
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
from moba import register_moba, MoBAConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
MOBA_AVAILABLE = True


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def _norm(self, x):
        """
        Applying RMS Normalization. Calculation is done in float32 for stability.
        """
        return (x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight

class MLASelfAttentionWithMoBA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert (config.n_embd // config.n_head) % 2 == 0, "Head dimension must be even for concatenation."

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_latent = config.d_latent
        self.d_head = config.n_embd // config.n_head
        
        self.d_head_e = self.d_head // 2 # "Embedding" part with RoPE
        self.d_head_c = self.d_head // 2 # "Context" part, plain
        
        self.dropout = config.dropout
        
        # MoBA configuration
        self.attn_implementation = getattr(config, 'attn_implementation', 'flash_attention_2')
        self.use_moba = 'moba' in self.attn_implementation if hasattr(config, 'attn_implementation') else False
        
        # Standard MLA projections
        self.q_down_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias=config.bias)
        self.k_down_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias=config.bias)
        self.v_down_proj = nn.Linear(self.n_embd, self.n_head * self.d_latent, bias=config.bias)

        self.q_up_proj_c = nn.Linear(self.d_latent, self.d_head_c, bias=config.bias)
        self.q_up_proj_e = nn.Linear(self.d_latent, self.d_head_e, bias=config.bias)
        self.k_up_proj_c = nn.Linear(self.d_latent, self.d_head_c, bias=config.bias)
        self.k_up_proj_e = nn.Linear(self.d_latent, self.d_head_e, bias=config.bias)
        
        self.v_up_proj = nn.Linear(self.d_latent, self.d_head, bias=config.bias)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
       
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.rope_cache = self.precompute_rope_cache(self.d_head_e, config.block_size)
        
        # Attention backend selection
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if self.use_moba and MOBA_AVAILABLE:
            print(f"✓ Using MoBA attention: {self.attn_implementation}")
        elif self.flash:
            print("✓ Using Flash Attention")
        else:
            print("⚠️  Using manual attention implementation")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def precompute_rope_cache(self, dim, max_seq_len, theta=10000.0):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

    def apply_rope(self, x, cos, sin):
        original_dtype = x.dtype
        device_type = x.device.type
        if device_type == 'mps':
            x = x.contiguous().to(memory_format=torch.contiguous_format)
        else:
            x = x.contiguous()
        
        x_f = x.to(torch.float32).contiguous()
        cos_f = cos[:x_f.shape[2]].to(torch.float32).unsqueeze(0).unsqueeze(0).contiguous()
        sin_f = sin[:x_f.shape[2]].to(torch.float32).unsqueeze(0).unsqueeze(0).contiguous()
        
        # Reshape with explicit contiguous calls
        B, H, T, C = x_f.shape
        x_reshaped = x_f.view(B, H, T, C // 2, 2).contiguous()
        x1, x2 = x_reshaped.unbind(-1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # Rotate with explicit contiguous calls
        rotated_x = torch.stack((-x2, x1), dim=-1).contiguous()
        rotated_x = rotated_x.view(B, H, T, C).contiguous()

        out_f = (x_f * cos_f) + (rotated_x * sin_f)
        out_f = out_f.contiguous()

        # Cast the final result back to the original dtype and ensure contiguity
        result = out_f.to(original_dtype).contiguous()
        
        if device_type == 'mps': 
            result = result.to(memory_format=torch.contiguous_format)
            
        return result

    def forward(self, x, past_kv=None):
        B, T, C = x.size()
        device_type = x.device.type

        # Ensure input tensor is contiguous and properly formatted for MPS
        if device_type == 'mps':
            x = x.contiguous().to(memory_format=torch.contiguous_format)
        else:
            x = x.contiguous()

        # MLA attention computation (same as original)
        q_latent = self.q_down_proj(x).view(B, T, self.n_head, self.d_latent).contiguous()
        k_latent_new = self.k_down_proj(x).view(B, T, self.n_head, self.d_latent).contiguous()
        v_latent_new = self.v_down_proj(x).view(B, T, self.n_head, self.d_latent).contiguous()

        if past_kv is not None:
            past_k, past_v = past_kv
            k_latent = torch.cat((past_k, k_latent_new), dim=1).contiguous()
            v_latent = torch.cat((past_v, v_latent_new), dim=1).contiguous()
        else:
            k_latent = k_latent_new
            v_latent = v_latent_new
        
        present_kv = (k_latent, v_latent)
        
        q_c = self.q_up_proj_c(q_latent).transpose(1, 2).contiguous()
        q_e = self.q_up_proj_e(q_latent).transpose(1, 2).contiguous()
        k_c = self.k_up_proj_c(k_latent).transpose(1, 2).contiguous()
        k_e = self.k_up_proj_e(k_latent).transpose(1, 2).contiguous()
        
        v = self.v_up_proj(v_latent).transpose(1, 2).contiguous()
        
        cos, sin = self.rope_cache
        cos, sin = cos.to(q_e.device), sin.to(q_e.device)
        
        q_e_rope = self.apply_rope(q_e, cos, sin)
        k_e_rope = self.apply_rope(k_e, cos, sin)

        q = torch.cat([q_c, q_e_rope], dim=-1).contiguous()
        k = torch.cat([k_c, k_e_rope], dim=-1).contiguous()

        # Attention computation with MoBA support
        if self.use_moba and MOBA_AVAILABLE and self.attn_implementation in ALL_ATTENTION_FUNCTIONS:
            # Use MoBA attention
            moba_fn = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
            y = moba_fn(
                query=q,
                key=k, 
                value=v,
                attention_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        elif self.flash:
            # Use Flash Attention
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Ensure MPS compatibility for output
        if device_type == 'mps':
            y = y.to(memory_format=torch.contiguous_format)
            
        y = self.resid_dropout(self.c_proj(y))
        
        return y, present_kv

# Alias for backward compatibility
MLASelfAttention = MLASelfAttentionWithMoBA

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
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

class MixtureOfExperts(nn.Module):
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
        
        self.load_balancing_bias = nn.Parameter(torch.zeros(self.n_routed_experts))
        self.load_balancing_bias.requires_grad = False

    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.view(-1, C).contiguous()
        n_tokens = x_reshaped.shape[0]

        shared_output = torch.zeros_like(x_reshaped).contiguous()
        for i in range(self.n_shared_experts):
            shared_output += self.experts[i](x_reshaped)

        routed_logits = self.gate(x_reshaped)
        selection_logits = routed_logits + self.load_balancing_bias
        
        top_k_indices_routed = torch.topk(selection_logits, self.n_routed_experts_per_token, dim=-1).indices

        routed_probs = F.softmax(routed_logits, dim=-1, dtype=torch.float32)
        top_k_weights = torch.gather(routed_probs, -1, top_k_indices_routed)

        if self.training:
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_indices_routed, num_classes=self.n_routed_experts)
                load_fraction = expert_mask.float().sum(dim=(0, 1)) / (n_tokens * self.n_routed_experts_per_token)
                ideal_load = 1.0 / self.n_routed_experts
                update = (load_fraction - ideal_load) * self.config.bias_update_gamma
                self.load_balancing_bias.data -= update

        top_k_indices = top_k_indices_routed + self.n_shared_experts
        routed_output = torch.zeros_like(x_reshaped).contiguous()
        flat_top_k_indices = top_k_indices.view(-1)
        repeated_x = x_reshaped.repeat_interleave(self.n_routed_experts_per_token, dim=0).contiguous()

        expert_outputs = torch.empty_like(repeated_x).contiguous()
        for i in range(self.n_shared_experts, self.n_experts):
            mask = (flat_top_k_indices == i)
            if mask.any():
                expert_outputs[mask] = self.experts[i](repeated_x[mask]).to(expert_outputs.dtype)

        weighted_outputs = (expert_outputs * top_k_weights.view(-1, 1)).contiguous()
        original_indices = torch.arange(n_tokens, device=x.device).repeat_interleave(self.n_routed_experts_per_token)
        routed_output.index_add_(0, original_indices, weighted_outputs)

        final_output = (shared_output + routed_output).contiguous()
        return final_output.view(B, T, C), None

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd) 
        self.attn = MLASelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) 
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
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True
    
    n_main_layers: int = 8
    n_branch_layers: int = 2
    n_mtp_branches: int = 2
    n_tokens_per_branch: tuple = (1, 1)

    n_head: int = 8
    d_latent: int = 64
    n_experts: int = 8
    n_experts_per_token: int = 4
    n_shared_experts: int = 2
    mlp_expansion_factor: int = 2
    bias_update_gamma: float = 1e-2
    
    # MoBA configuration
    attn_implementation: str = "moba"  # Can be "flash_attention_2", "moba", "moba_naive"
    moba_chunk_size: int = 4096
    moba_topk: int = 12

    @property
    def n_layer(self):
        return self.n_main_layers + self.n_branch_layers

    @property
    def n_predicted_tokens(self):
        return sum(self.n_tokens_per_branch)

def setup_moba(config):
    """Setup MoBA if requested and available"""
    if MOBA_AVAILABLE and 'moba' in config.attn_implementation:
        moba_config = MoBAConfig(
            moba_chunk_size=config.moba_chunk_size,
            moba_topk=config.moba_topk
        )
        register_moba(moba_config)
        print(f"✓ MoBA registered with chunk_size={config.moba_chunk_size}, topk={config.moba_topk}")
        return True
    return False

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Setup MoBA if requested
        setup_moba(config)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h_main = nn.ModuleList([Block(config) for _ in range(config.n_main_layers)]),
            h_branches = nn.ModuleList([
                nn.ModuleList([Block(config) for _ in range(config.n_branch_layers)])
                for _ in range(config.n_mtp_branches)
            ]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.n_mtp_branches)
        ])
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

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

    def forward(self, input_ids, labels=None):
        # Same forward implementation as original
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        # 1. Main Backbone Pass (runs once)
        tok_emb = self.transformer.wte(input_ids)
        backbone_h = self.transformer.drop(tok_emb)
        
        for block in self.transformer.h_main:
            backbone_h, _ = block(backbone_h)

        all_logits = []
        loss = None

        if labels is not None:
            # Training mode - same as original implementation
            current_h = backbone_h
            
            for i, branch_blocks in enumerate(self.transformer.h_branches):
                branch_h = current_h
                for block in branch_blocks:
                    branch_h, _ = block(branch_h)

                final_h = self.transformer.ln_f(branch_h)
                logits = self.lm_heads[i](final_h)
                all_logits.append(logits)
                
                if i < len(self.transformer.h_branches) - 1:
                    if logits.size(1) > 0:
                        next_token_probs = F.softmax(logits[:, -1, :], dim=-1)
                        
                        if i == 0:
                            if T < labels.size(1):
                                next_token_ids = labels[:, T:T+1]
                            else:
                                next_token_ids = torch.multinomial(next_token_probs, num_samples=1)
                        else:
                            token_offset = i + 1
                            if T + token_offset < labels.size(1):
                                next_token_ids = labels[:, T + token_offset:T + token_offset + 1]
                            else:
                                next_token_ids = torch.multinomial(next_token_probs, num_samples=1)
                        
                        next_token_emb = self.transformer.wte(next_token_ids)
                        current_h = torch.cat([branch_h, next_token_emb], dim=1)
                        
                        if current_h.size(1) > self.config.block_size:
                            current_h = current_h[:, -self.config.block_size:, :]
                    else:
                        current_h = branch_h

            # Calculate loss (same as original)
            if all_logits:
                device = input_ids.device
                total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
                loss_token_offset = 1
                for i in range(self.config.n_mtp_branches):
                    branch_logits = all_logits[i]
                    num_predicted = self.config.n_tokens_per_branch[i]
                    for j in range(num_predicted):
                        current_offset = loss_token_offset + j
                        current_logits = branch_logits[:, :-current_offset, :].contiguous()
                        current_targets = labels[:, current_offset:].contiguous()
                        
                        if current_logits.size(1) > 0 and current_targets.size(1) > 0:
                            min_len = min(current_logits.size(1), current_targets.size(1))
                            current_logits = current_logits[:, :min_len, :]
                            current_targets = current_targets[:, :min_len]
                            
                            l = F.cross_entropy(
                                current_logits.reshape(-1, current_logits.size(-1)),
                                current_targets.reshape(-1),
                                ignore_index=-100
                            )
                            total_loss = total_loss + l
                    loss_token_offset += num_predicted
                loss = total_loss / float(self.config.n_predicted_tokens)

        # For training, return the logits from the first branch for monitoring
        final_logits = all_logits[0] if all_logits else None
        if labels is None:
            current_h = backbone_h
            for i, branch_blocks in enumerate(self.transformer.h_branches):
                branch_h = current_h
                for block in branch_blocks:
                    branch_h, _ = block(branch_h)
                
                final_h = self.transformer.ln_f(branch_h)
                logits = self.lm_heads[i](final_h)
                
                if i == 0:
                    final_logits = logits
                
                break

        return final_logits, loss

    # Rest of the methods remain the same as original
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
    
        all_blocks = list(self.transformer.h_main)
        for branch in self.transformer.h_branches:
            all_blocks.extend(branch)

        for block in all_blocks:
            block.attn.rope_cache = block.attn.precompute_rope_cache(
                dim=block.attn.d_head_e,
                max_seq_len=self.config.block_size
            )
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            tok_emb = self.transformer.wte(idx_cond)
            current_h = self.transformer.drop(tok_emb)
            for block in self.transformer.h_main:
                current_h, _ = block(current_h)

            generated_tokens = []
            
            for i, branch_blocks in enumerate(self.transformer.h_branches):
                branch_h = current_h
                for block in branch_blocks:
                    branch_h, _ = block(branch_h)
                
                final_h = self.transformer.ln_f(branch_h)
                logits = self.lm_heads[i](final_h[:, -1, :])
                
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(idx_next)
                
                if i < len(self.transformer.h_branches) - 1:
                    next_token_emb = self.transformer.wte(idx_next)
                    current_h = torch.cat([branch_h, next_token_emb], dim=1)
                    
                    if current_h.size(1) > self.config.block_size:
                        current_h = current_h[:, -self.config.block_size:, :]

            for token in generated_tokens:
                idx = torch.cat((idx, token), dim=1)
                break

        return idx
