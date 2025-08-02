import torch
from muon_optimizer import muon_clip_update, adam_update


class SingleDeviceMuonClipWithAuxAdam(torch.optim.Optimizer):
    """
    MuonClip variant that can be used for all parameters in the network, 
    combining MuonClip for matrices and AdamW for other parameters.
    
    This implements the MuonClip algorithm from Kimi K2:
    1. Muon optimizer step with RMS scaling: sqrt(max(n,m)) * 0.2
    2. QK-Clip: gradient clipping for attention weights based on softmax max values
    """
    def __init__(self, param_groups, qk_clip_threshold=100.0):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults for MuonClip
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["qk_clip_threshold"] = group.get("qk_clip_threshold", qk_clip_threshold)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "qk_clip_threshold"])
            else:
                # defaults for AdamW
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())
        
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
            if group["use_muon"]:
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
                    
                    # Only apply MuonClip to 2D+ parameters (matrices)
                    if p.grad.ndim >= 2:
                        update = muon_clip_update(grad, state["momentum_buffer"], beta=group["momentum"])
                        # Handle reshape carefully
                        if update.shape != p.shape:
                            if update.numel() == p.numel():
                                update = update.reshape(p.shape)
                            else:
                                # Fallback: use original gradient
                                update = grad
                    else:
                        state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
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
    
    def _apply_qk_clip(self, grad, param_name, threshold):
        """Apply QK-Clip based on stored attention max values"""
       
        for key, max_val in self.attention_max_values.items():
            if any(name in param_name.lower() for name in key.split('_')):
                if max_val > threshold:
                    # Compute clipping factor: γ = τ / S_max^h
                    gamma = threshold / max_val
                    sqrt_gamma = torch.sqrt(torch.tensor(gamma, device=grad.device))
                    grad = grad * sqrt_gamma
                break
        
        return grad
