"""
Full definition of a GPT Language Model, all of it in this single file.
Really it is a GPT Model but with DeepSeek modifications
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# In mla_model.py

class MLASelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_latent = config.d_latent
        self.d_head = config.n_embd // config.n_head
        self.dropout = config.dropout

        # MLA: separted projection layer
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias = config.bias)
        self.k_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias = config.bias)
        self.v_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias = config.bias)
        self.k_up_proj = nn.Linear(self.d_latent, self.d_head, bias = config.bias)
        self.v_up_proj = nn.Linear(self.d_latent, self.d_head, bias = config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias= config.bias)

        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.rope_cache = self.precompute_rope_cache(self.d_head, config.block_size)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def precompute_rope_cache(self, dim, max_seq_len, theta=10000.0):
        # Generates the sine and cosine components for RoPE. These are fixed and can be cached.
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Returns cos(m*theta_i), sin(m*theta_i)
        return torch.cos(emb), torch.sin(emb)

    def apply_rope(self, x, cos, sin):
        # Applies the RoPE rotation to the input tensor x (query or key).
        B, H, T, C = x.shape
        cos = cos[:T].unsqueeze(0).unsqueeze(0) # (1, 1, T, C)
        sin = sin[:T].unsqueeze(0).unsqueeze(0) # (1, 1, T, C)
        
        # Reshape x to easily apply rotation to pairs of dimensions
        x_reshaped = x.float().reshape(B, H, T, C // 2, 2)
        x1, x2 = x_reshaped.unbind(-1)
        
        # Apply the rotation matrix
        rotated_x = torch.stack((-x2, x1), dim=-1).flatten(-2)
        
        return (x * cos + rotated_x * sin).to(x.dtype)

    def forward(self, x, past_kv = None):
        B, T, C = x.size()

        # Latent space projections for K and V
        k_latent_new = self.k_proj(x).view(B, T, self.n_head, self.d_latent)
        v_latent_new = self.v_proj(x).view(B, T, self.n_head, self.d_latent)

        if past_kv is not None:
            past_k, past_v = past_kv
            k_latent = torch.cat((past_k, k_latent_new), dim=1)
            v_latent = torch.cat((past_v, v_latent_new), dim=1)
        else:
            k_latent = k_latent_new
            v_latent = v_latent_new
        
        present_kv = (k_latent, v_latent)
        
        # Project Q, K, V to head dimension
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_up_proj(k_latent).transpose(1, 2)
        v = self.v_up_proj(v_latent).transpose(1, 2)

       
        cos, sin = self.rope_cache
        cos = cos.to(q.device)
        sin = sin.to(q.device)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)
       

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv


class MixtureOfExperts(nn.Module):
    """
    A Mixture of Experts layer implementing Shared Expert Isolation and
    Auxiliary-Loss-Free Load Balancing.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_experts_per_token > config.n_shared_experts
        self.config = config
        self.n_experts = config.n_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_routed_experts = self.n_experts - self.n_shared_experts
        self.n_routed_experts_per_token = config.n_experts_per_token - config.n_shared_experts

        self.gate = nn.Linear(config.n_embd, self.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.n_experts)])
        
        #  Add the learnable bias for load balancing ---
        # This parameter is manually updated, not through backpropagation.
        self.load_balancing_bias = nn.Parameter(torch.zeros(self.n_routed_experts))
        self.load_balancing_bias.requires_grad = False

    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)
        n_tokens = x_reshaped.shape[0]

        #  Process Shared Experts (same as before) ---
        shared_output = torch.zeros_like(x_reshaped)
        for i in range(self.n_shared_experts):
            shared_output += self.experts[i](x_reshaped)

        # Loss-Free Load Balancing Logic ---
        # 1. Get routed logits and apply bias for selection
        routed_logits = self.gate(x_reshaped)
        selection_logits = routed_logits + self.load_balancing_bias
        
        # 2. Perform Top-K selection on the biased logits
        top_k_indices_routed = torch.topk(selection_logits, self.n_routed_experts_per_token, dim=-1).indices

        # 3. Calculate gating weights from the ORIGINAL, unbiased logits
        routed_probs = F.softmax(routed_logits, dim=-1, dtype=torch.float32)
        # Gather the probabilities of the chosen experts
        top_k_weights = torch.gather(routed_probs, -1, top_k_indices_routed)
        # This is not re-normalized, as per the paper's formulation

        # --- 4. Manually update the load-balancing bias during training ---
        if self.training:
            with torch.no_grad():
                # Get a one-hot mask of which experts were chosen for each token
                expert_mask = F.one_hot(top_k_indices_routed, num_classes=self.n_routed_experts)
                # Calculate the load fraction for each expert (how often it was chosen)
                load_fraction = expert_mask.float().sum(dim=(0, 1)) / (n_tokens * self.n_routed_experts_per_token)
                # The ideal load is uniform
                ideal_load = 1.0 / self.n_routed_experts
                
                # Calculate the update amount
                update = (load_fraction - ideal_load) * self.config.bias_update_gamma
                
                # Update the bias: decrease for overloaded, increase for underloaded
                self.load_balancing_bias.data -= update

        # --- Dispatch to routed experts (same as before, but no aux_loss) ---
        top_k_indices = top_k_indices_routed + self.n_shared_experts
        routed_output = torch.zeros_like(x_reshaped)
        flat_top_k_indices = top_k_indices.view(-1)
        repeated_x = x_reshaped.repeat_interleave(self.n_routed_experts_per_token, dim=0)

        expert_outputs = torch.empty_like(repeated_x)
        for i in range(self.n_shared_experts, self.n_experts):
            mask = (flat_top_k_indices == i)
            if mask.any():
                expert_outputs[mask] = self.experts[i](repeated_x[mask]).to(expert_outputs.dtype)

        weighted_outputs = expert_outputs * top_k_weights.view(-1, 1)
        original_indices = torch.arange(n_tokens, device=x.device).repeat_interleave(self.n_routed_experts_per_token)
        routed_output.index_add_(0, original_indices, weighted_outputs)

        final_output = shared_output + routed_output
        
        # --- MODIFICATION: Return None for the auxiliary loss ---
        return final_output.view(B, T, C), None

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Cast the hidden dimension to an integer
        hidden_dim = int(config.n_embd * config.mlp_expansion_factor)
        
        self.c_fc    = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MLASelfAttention(config) # Your existing attention module
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MixtureOfExperts(config)

    
    def forward(self, x, past_kv=None):
        attn_output, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_output
        mlp_output, _ = self.mlp(self.ln_2(x))
        x = x + mlp_output
        
       
        return x, present_kv

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    d_latent: int = 64 # set it smaller
    dropout: float = 0.0
    bias: bool = True
    n_experts: int = 8
    n_experts_per_token: int = 4
    n_shared_experts: int = 2
    mlp_expansion_factor: int = 2
    # --- NEW: Add bias update gamma and remove aux_loss_weight ---
    bias_update_gamma: float = 1e-2 # Learning rate for the load balancing bias
    # aux_loss_weight is no longer needed

    n_shared_experts: int = 2 # Number of experts to always activate

    #n_eexperts_per_token > n_shared_experts 

    # n_expert_groups is no longer needed for this approach
    aux_loss_weight: float = 1e-4
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe is removed because RoPE is used in the attention blocks
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        total_params = self.get_num_params()
        print(f"number of parameters: {total_params/1e6:.2f}M")

        # if it's an MoE model, print detailed stats
        if hasattr(config, 'n_experts') and config.n_experts > 0:
            # calculate params for one MLP expert and one full MoE layer
            mlp_params = sum(p.numel() for p in MLP(config).parameters())
            moe_params = sum(p.numel() for p in MixtureOfExperts(config).parameters())
            
            # calculate active params: non-expert params + params for active experts
            non_expert_moe_params = moe_params - (config.n_experts * mlp_params)
            active_moe_params = non_expert_moe_params + (config.n_experts_per_token * mlp_params)

            print("---")
            print(f"MoE Configuration:")
            print(f"  - MoE Layers: {config.n_layer}")
            print(f"  - Params per MoE Layer: {moe_params:,} (~{moe_params/1e6:.2f}M)")
            print(f"  - Active Params per MoE Layer: {int(active_moe_params):,} (~{active_moe_params/1e6:.2f}M)")
            print("---")
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
       

        if non_embedding:
            # We no longer subtract wpe, because it doesn't exist.
            pass
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # In mla_model.py, inside the GPT class

    def forward(self, idx, targets=None, past_kv_cache=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        is_inference = past_kv_cache is not None
        if not is_inference:
            past_kv_cache = [None] * len(self.transformer.h)
        
        present_kv_cache = []
        x = self.transformer.drop(tok_emb)
        
       
        for i, block in enumerate(self.transformer.h):
            # The block's forward call is now simpler
            x, present_kv = block(x, past_kv=past_kv_cache[i])
            present_kv_cache.append(present_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            
            logits = self.lm_head(x)
            # --- MODIFICATION: Loss is now a single value ---
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if is_inference:
            # If we are generating, return the cache
            return logits, loss, present_kv_cache
        else:
            # --- MODIFICATION: Return the single loss tensor, not a tuple ---
            return logits, loss
        

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        # With RoPE, we don't crop a weight table. Instead, we recompute
        # the RoPE cache in each attention block with the new block size.
        for block in self.transformer.h:
            # Recompute the RoPE cache with the new block_size
            block.attn.rope_cache = block.attn.precompute_rope_cache(
                dim=block.attn.d_head,
                max_seq_len=self.config.block_size
            )
            # The logic for the causal mask 'bias' remains the same
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        kv_cache = None
        for _ in range(max_new_tokens):
            idx_cond = idx if kv_cache is None else idx[:, -1:]

            # fwd pass model 
            logits, _, kv_cache = self(idx_cond, past_kv_cache=kv_cache)

            logits = logits[:, -1,:] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx