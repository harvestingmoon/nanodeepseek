
import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() if hasattr(G, 'bfloat16') else G.float()
    
    # Ensure all tensors are on the same device and contiguous
    device = X.device
    X = X.contiguous()
    
    if G.size(-2) > G.size(-1):
        X = X.mT.contiguous()

    # Ensure spectral norm is at most 1
    norm_val = X.norm(dim=(-2, -1), keepdim=True) + eps
    X = (X / norm_val).contiguous()

    for _ in range(steps):
        A = (X @ X.mT).contiguous()
        A = A.to(device)
        A_squared = (A @ A).contiguous()
        B = (b * A + c * A_squared).contiguous()
        X = (a * X + B @ X).contiguous()
    
    if G.size(-2) > G.size(-1):
        X = X.mT.contiguous()
    return X.to(G.dtype)


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    # Ensure all tensors are on the same device and contiguous
    device = grad.device
    grad = grad.contiguous()
    momentum = momentum.contiguous()
    
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = update.contiguous()


def muon_clip_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, use_rms_scaling=True):
    """
    MuonClip update function that implements the algorithm from Kimi K2.
    This includes RMS scaling factor matching Adam RMS: sqrt(max(n,m)) * 0.2
    """
    # Ensure all tensors are on the same device and contiguous
    device = grad.device
    grad = grad.contiguous()
    momentum = momentum.contiguous()
    
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = update.contiguous()
    
    # Handle convolutional filters
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1).contiguous()
    
    # Ensure update is 2D for Newton-Schulz
    if update.ndim == 1:
        return update  
    
    # Apply Newton-Schulz orthogonalization
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if use_rms_scaling:
        n, m = update.size(-2), update.size(-1)
        rms_scale_factor = (max(n, m) ** 0.5) * 0.2
        update = update * rms_scale_factor
    else:
        scale_factor = max(1, grad.size(-2) / grad.size(-1))**0.5
        update = update * scale_factor
    
    return update.to(device)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            # Handle non-distributed case
            if not dist.is_initialized():
                for p in params:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # Original distributed code
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class MuonClip(torch.optim.Optimizer):
    """
    MuonClip - Muon optimizer with QK-Clip as implemented in Kimi K2
    
    This combines:
    1. Muon optimizer step with RMS scaling: sqrt(max(n,m)) * 0.2
    2. QK-Clip: gradient clipping for attention weights based on softmax max values
    
    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
        qk_clip_threshold: Threshold τ for QK-Clip (default: 100.0)
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, qk_clip_threshold=100.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, qk_clip_threshold=qk_clip_threshold)
        super().__init__(params, defaults)
        
        # Store attention softmax max values for QK-Clip
        self.attention_max_values = {}

    def register_attention_max(self, layer_name, head_idx, softmax_max_val):
        """Register the max softmax value for QK-Clip"""
        key = f"{layer_name}_head_{head_idx}"
        self.attention_max_values[key] = softmax_max_val

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                
                # Apply QK-Clip if this is an attention parameter
                grad = p.grad
                param_name = getattr(p, '_param_name', '')
                
                if any(name in param_name.lower() for name in ['q_proj', 'k_proj', 'query', 'key']):
                    grad = self._apply_qk_clip(grad, param_name, group["qk_clip_threshold"])
                
                # Apply MuonClip update (Muon with RMS scaling)
                if grad.ndim >= 2:
                    update = muon_clip_update(grad, state["momentum_buffer"], beta=group["momentum"])
                else:
                    # For 1D parameters, just use momentum
                    state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                    update = state["momentum_buffer"]
                
                # Apply weight decay and learning rate
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
    
    def _apply_qk_clip(self, grad, param_name, threshold):
        """Apply QK-Clip based on stored attention max values"""
        # Try to find corresponding attention max value
        for key, max_val in self.attention_max_values.items():
            if any(name in param_name.lower() for name in key.split('_')):
                if max_val > threshold:
                    # Compute clipping factor: γ = τ / S_max^h
                    gamma = threshold / max_val
                    sqrt_gamma = torch.sqrt(torch.tensor(gamma, device=grad.device))
                    grad = grad * sqrt_gamma
                break
        
        return grad


class SingleDeviceMuonClip(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonClip for single device usage.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, qk_clip_threshold=100.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, qk_clip_threshold=qk_clip_threshold)
        super().__init__(params, defaults)
        
        # Store attention softmax max values for QK-Clip  
        self.attention_max_values = {}

    def register_attention_max(self, layer_name, head_idx, softmax_max_val):
        """Register the max softmax value for QK-Clip"""
        key = f"{layer_name}_head_{head_idx}"
        self.attention_max_values[key] = softmax_max_val

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Ensure grad is on the same device as parameter
                if p.grad.device != p.device:
                    p.grad = p.grad.to(p.device)
                
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, device=p.device)
                
                # Apply QK-Clip if this is an attention parameter
                grad = p.grad
                param_name = getattr(p, '_param_name', '')
                
                if any(name in param_name.lower() for name in ['q_proj', 'k_proj', 'query', 'key']):
                    grad = self._apply_qk_clip(grad, param_name, group["qk_clip_threshold"])
                
                # Apply MuonClip update
                if grad.ndim >= 2:
                    update = muon_clip_update(grad, state["momentum_buffer"], beta=group["momentum"])
                    # Handle reshape carefully
                    if update.shape != p.shape:
                        if update.numel() == p.numel():
                            update = update.reshape(p.shape)
                        else:
                            # Fallback: use original gradient
                            update = grad
                else:
                    # For 1D parameters (biases), just use momentum
                    state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                    update = state["momentum_buffer"]
                
                # Ensure update is on correct device and has correct shape
                update = update.to(p.device)
                if update.shape != p.shape:
                    update = update.reshape(p.shape)
                
                # Apply weight decay and learning rate
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])

        return loss
    
    def _apply_qk_clip(self, grad, param_name, threshold):
        """Apply QK-Clip based on stored attention max values"""
        # Try to find corresponding attention max value
        for key, max_val in self.attention_max_values.items():
            if any(name in param_name.lower() for name in key.split('_')):
                if max_val > threshold:
                    # Compute clipping factor: γ = τ / S_max^h  
                    gamma = threshold / max_val
                    sqrt_gamma = torch.sqrt(torch.tensor(gamma, device=grad.device))
                    grad = grad * sqrt_gamma
                break
        
        return grad


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                # Handle non-distributed case
                if not dist.is_initialized():
                    for p in params:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                else:
                    # Original distributed code
                    params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                    for base_i in range(len(params))[::dist.get_world_size()]:
                        if base_i + dist.get_rank() < len(params):
                            p = params[base_i + dist.get_rank()]
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)  # Force synchronization
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)
                            update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.add_(update.reshape(p.shape), alpha=-group["lr"])
                        dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    
                    if p.grad.device != p.device:
                        p.grad = p.grad.to(p.device)
                    
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p, device=p.device)
                    
                    
                    if p.grad.ndim >= 2:
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        # Handle reshape carefully
                        if update.shape != p.shape:
                            if update.numel() == p.numel():
                                update = update.reshape(p.shape)
                            else:
                                # Fallback: use original gradient
                                update = p.grad
                    else:
                        # For 1D parameters (biases), just use momentum without Newton-Schulz
                        state["momentum_buffer"].lerp_(p.grad, 1 - group["momentum"])
                        update = state["momentum_buffer"]
                    
                    # Ensure update is on correct device and has correct shape
                    update = update.to(p.device)
                    if update.shape != p.shape:
                        update = update.reshape(p.shape)
                        
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
