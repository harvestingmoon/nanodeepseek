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

    def __init__(self, ndim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def _norm(self, x): 
        """ 
            Applying RMS Normalization
        """
        # (B, T, C) * (B, T, 1) = (B, T, C)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

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

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MLASelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert (config.n_embd // config.n_head) % 2 == 0, "Head dimension must be even for concatenation."

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_latent = config.d_latent
        self.d_head = config.n_embd // config.n_head
        
        # The dimension of the RoPE-applied part and the plain part.
        # Wsplit the head dimension in two.
        self.d_head_e = self.d_head // 2 # "Embedding" part with RoPE
        self.d_head_c = self.d_head // 2 # "Context" part, plain
        
        self.dropout = config.dropout

        # --- Down-projection layers (h_t -> c_t) ---
        self.q_down_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias=config.bias)
        self.k_down_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias=config.bias)
        self.v_down_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias=config.bias)

        # --- Up-projection layers (c_t -> q, k, v) ---
        # Separate up-projections for the plain and RoPE-applied components
        self.q_up_proj_c = nn.Linear(self.d_latent, self.d_head_c, bias=config.bias)
        self.q_up_proj_e = nn.Linear(self.d_latent, self.d_head_e, bias=config.bias)
        self.k_up_proj_c = nn.Linear(self.d_latent, self.d_head_c, bias=config.bias)
        self.k_up_proj_e = nn.Linear(self.d_latent, self.d_head_e, bias=config.bias)
        
        # Value path does not have concatenation or RoPE
        self.v_up_proj = nn.Linear(self.d_latent, self.d_head, bias=config.bias)

        # --- Final output projection ---
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        # --- Regularization ---
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

        # RoPE cache is now computed for the smaller RoPE dimension
        self.rope_cache = self.precompute_rope_cache(self.d_head_e, config.block_size)
        
        # Flash Attention check
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def precompute_rope_cache(self, dim, max_seq_len, theta=10000.0):
        # This function remains the same
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

    def apply_rope(self, x, cos, sin):
        B, H, T, C = x.shape
        cos = cos[:T].unsqueeze(0).unsqueeze(0)
        sin = sin[:T].unsqueeze(0).unsqueeze(0)
        x_reshaped = x.float().reshape(B, H, T, C // 2, 2)
        x1, x2 = x_reshaped.unbind(-1)
        rotated_x = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos + rotated_x * sin).to(x.dtype)

    def forward(self, x, past_kv=None):
        B, T, C = x.size()

        # 1. Down-project to latent space for Q, K, V
        q_latent = self.q_down_proj(x).view(B, T, self.n_head, self.d_latent)
        k_latent_new = self.k_down_proj(x).view(B, T, self.n_head, self.d_latent)
        v_latent_new = self.v_down_proj(x).view(B, T, self.n_head, self.d_latent)

        # Handle KV cache for inference
        if past_kv is not None:
            past_k, past_v = past_kv
            k_latent = torch.cat((past_k, k_latent_new), dim=1)
            v_latent = torch.cat((past_v, v_latent_new), dim=1)
        else:
            k_latent = k_latent_new
            v_latent = v_latent_new
        
        present_kv = (k_latent, v_latent)
        
        # 2. Up-project from latent space
        # For Q and K, create plain (_c) and RoPE-able (_e) components
        q_c = self.q_up_proj_c(q_latent).transpose(1, 2) # (B, nh, T, d_head_c)
        q_e = self.q_up_proj_e(q_latent).transpose(1, 2) # (B, nh, T, d_head_e)
        k_c = self.k_up_proj_c(k_latent).transpose(1, 2) # (B, nh, T, d_head_c)
        k_e = self.k_up_proj_e(k_latent).transpose(1, 2) # (B, nh, T, d_head_e)
        
        # Value is a up-projection
        v = self.v_up_proj(v_latent).transpose(1, 2)     # (B, nh, T, d_head)
        
        # 3. Apply RoPE to the '_e' components
        cos, sin = self.rope_cache
        cos, sin = cos.to(q_e.device), sin.to(q_e.device)
        
        q_e_rope = self.apply_rope(q_e, cos, sin)
        k_e_rope = self.apply_rope(k_e, cos, sin)

        # 4. Concatenate plain and RoPE-applied parts to form final Q and K
        q = torch.cat([q_c, q_e_rope], dim=-1)
        k = torch.cat([k_c, k_e_rope], dim=-1)

        # 5. Perform attention
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # 6. Final projection
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

        #  Process Shared Experts (same as before) 
        shared_output = torch.zeros_like(x_reshaped)
        for i in range(self.n_shared_experts):
            shared_output += self.experts[i](x_reshaped)

        # Loss-Free Load Balancing Logic 

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

        #  4. Manually update the load-balancing bias during training 
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

        # Dispatch to routed experts (same as before, but no aux_loss)
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
        

        # aux loss returns none
        return final_output.view(B, T, C), None



# Dont even need it anymore lol``
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
        self.ln_1 = LayerNorm(config.n_embd) 
        self.attn = MLASelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd) 
        self.mlp = MixtureOfExperts(config)

    
    def forward(self, x, past_kv=None):
        attn_output, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_output
        mlp_output, _ = self.mlp(self.ln_2(x))
        x = x + mlp_output
        return x, present_kv

@dataclass
class GPTConfig:
    block_size: int = 1024 # effectively this is the sequence length
    vocab_size: int = 50304
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True
    
    # --- MTP Architecture Config ---
    n_main_layers: int = 8          # Number of layers in the shared backbone
    n_branch_layers: int = 2        # Number of layers in each MTP branch
    n_mtp_branches: int = 2         # Number of parallel MTP branches


    # Defines how many future tokens each branch predicts. 
    # e.g., [1, 1] means branch 0 predicts 1 token, branch 1 predicts 1 token.
    # Total predicted tokens per step = sum(n_tokens_per_branch)
    n_tokens_per_branch: tuple = (1, 1)

    # --- Standard Transformer/MoE Config (as before) ---
    n_head: int = 8
    d_latent: int = 64
    n_experts: int = 8
    n_experts_per_token: int = 4
    n_shared_experts: int = 2
    mlp_expansion_factor: int = 2
    bias_update_gamma: float = 1e-2

    # --- Calculated properties for convenience ---
    @property
    def n_layer(self):
        return self.n_main_layers + self.n_branch_layers

    @property
    def n_predicted_tokens(self):
        # Total number of tokens predicted in one forward pass
        return sum(self.n_tokens_per_branch)
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h_main = nn.ModuleList([Block(config) for _ in range(config.n_main_layers)]),
            h_branches = nn.ModuleList([
                nn.ModuleList([Block(config) for _ in range(config.n_branch_layers)])
                for _ in range(config.n_mtp_branches)
            ]),
            ln_f = LayerNorm(config.n_embd),
        ))
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.n_mtp_branches)
        ])
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    # __init__ helper methods (get_num_params, _init_weights) remain the same...
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    # In mla_model.py, inside the GPT class

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        # 1. Main Backbone Pass
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h_main:
            x, _ = block(x)

        # This is the initial hidden state for the MTP chain
        current_h = x
        all_logits = []
        loss = None

        # This block handles both training (if targets is not None) and inference
        if targets is not None:
            token_offset = 1
            for i, branch_blocks in enumerate(self.transformer.h_branches):
                branch_h = current_h
                for block in branch_blocks:
                    branch_h, _ = block(branch_h)

                final_h = self.transformer.ln_f(branch_h)
                logits = self.lm_heads[i](final_h)
                all_logits.append(logits)

                # Causal Link: Condition the next branch on the ground-truth token
                if i < self.config.n_mtp_branches - 1:
                    num_predicted = self.config.n_tokens_per_branch[i]
                    next_target_tokens = targets[:, token_offset : token_offset + num_predicted].contiguous()
                    next_target_emb = self.transformer.wte(next_target_tokens)
                    T_next = next_target_emb.shape[1]
                    # Fuse the branch output with the ground-truth embedding
                    current_h = branch_h[:, :T_next, :] + next_target_emb
                    token_offset += num_predicted

            # --- Correctly Integrated Loss Calculation ---
            if all_logits: # Only calculate loss if logits were generated
                total_loss = 0
                loss_token_offset = 1
                for i in range(self.config.n_mtp_branches):
                    branch_logits = all_logits[i]
                    num_predicted = self.config.n_tokens_per_branch[i]
                    for j in range(num_predicted):
                        current_offset = loss_token_offset + j
                        # This slicing is now safe
                        current_logits = branch_logits[:, :-current_offset, :].contiguous()
                        current_targets = targets[:, current_offset:].contiguous()
                        
                        # This check prevents the error
                        if current_logits.size(1) > 0 and current_targets.size(1) > 0:
                            l = F.cross_entropy(
                                current_logits.view(-1, current_logits.size(-1)),
                                current_targets.view(-1),
                                ignore_index=-1
                            )
                            total_loss += l
                    loss_token_offset += num_predicted
                loss = total_loss / self.config.n_predicted_tokens

        # For training, we can return the first set of logits for monitoring purposes
        final_logits = all_logits[0] if all_logits else None

        # This path is for inference, where targets=None
        if targets is None:
            # We only need to run the first branch to generate the next token
            branch_h = current_h
            for block in self.transformer.h_branches[0]:
                branch_h, _ = block(branch_h)
            final_h = self.transformer.ln_f(branch_h)
            final_logits = self.lm_heads[0](final_h)

        return final_logits, loss
        

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
    # @classmethod
    # def from_pretrained(cls, model_type, override_args=None):
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     override_args = override_args or {} # default to empty dict
    #     # only dropout can be overridden see more notes below
    #     assert all(k == 'dropout' for k in override_args)
    #     from transformers import GPT2LMHeadModel
    #     print("loading weights from pretrained gpt: %s" % model_type)

    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    #         'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    #         'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    #         'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    #     }[model_type]
    #     print("forcing vocab_size=50257, block_size=1024, bias=True")
    #     config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    #     config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    #     config_args['bias'] = True # always True for GPT model checkpoints
    #     # we can override the dropout rate, if desired
    #     if 'dropout' in override_args:
    #         print(f"overriding dropout rate to {override_args['dropout']}")
    #         config_args['dropout'] = override_args['dropout']
    #     # create a from-scratch initialized minGPT model
    #     config = GPTConfig(**config_args)
    #     model = GPT(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    #     # this means that we have to transpose these weights when we import them
    #     assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    #     for k in sd_keys_hf:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model

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
        # since we are using 3060 RTX Mobile ( it is roughly 12.74 TFLOPS)

        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 12.74e12 
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generation with a Causal Chain MTP. This is more complex than standard
        autoregressive decoding because we must manually create the causal chain
        at each time step.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # One full MTP Prediction
            
            # 1. Get backbone hidden state
            tok_emb = self.transformer.wte(idx_cond)
            current_h = self.transformer.drop(tok_emb)
            for block in self.transformer.h_main:
                current_h, _ = block(current_h)

            # 2. Sequentially run the MTP branches
            for i, branch_blocks in enumerate(self.transformer.h_branches):
                # We only care about the *last* token's representation for generation
                # but we process the whole sequence for the KV cache to be correct.
                branch_h = current_h
                for block in branch_blocks:
                    branch_h, _ = block(branch_h)
                
                # Get logits for the last token
                final_h = self.transformer.ln_f(branch_h)
                logits = self.lm_heads[i](final_h[:, -1, :]) # Shape: (B, Vocab_Size)
                
                # Apply sampling strategy (temp, top-k)
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                
                # Sample the next token
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append the newly generated token to the sequence
                idx = torch.cat((idx, idx_next), dim=1)
                if i < self.config.n_mtp_branches - 1:
                    # Get the embedding of the token we just sampled
                    next_token_emb = self.transformer.wte(idx_next) # Shape: (B, 1, C)
                    # Fuse it with the last hidden state of the current branch
                    current_h = branch_h[:, -1:, :] + next_token_emb

        self.train()
        return idx