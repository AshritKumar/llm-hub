"""
GPT Model with Grouped Query Attention (GQA) + RoPE + Mixture of Experts (MoE)

GQA changes from standard MHA:
    - Q keeps all n_heads independent projections            → (ED, n_heads * head_dim)
    - K and V share projections across groups of Q heads     → (ED, n_kv_heads * head_dim)
    - Before attention, K/V are expanded to match Q's head count via repeat_interleave
    - Everything else (attention math, output projection) is unchanged

This reduces KV cache by a factor of (n_heads / n_kv_heads) with minimal quality loss.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================

@dataclass
class GQAMoEConfig:
    # ── Transformer ──────────────────────────────────────────────────────────
    vocab_size:           int   = 65
    context_len:          int   = 256
    emb_dim:              int   = 384    # ED
    n_heads:              int   = 8      # query heads  (must be divisible by n_kv_heads)
    n_kv_heads:           int   = 2      # key/value heads  (the GQA knob)
    n_transformer_blocks: int   = 6
    dropout:              float = 0.05
    # ── MoE ──────────────────────────────────────────────────────────────────
    n_experts:            int   = 8
    top_k:                int   = 2
    ffn_mult:             int   = 4
    aux_loss_coeff:       float = 0.01

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"


# =============================================================================
# RoPE  (from gpt_rope_model.py)
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., head_dim) — interleaved pairs
    x1 = x[..., ::2]    # (..., head_dim/2)
    x2 = x[..., 1::2]   # (..., head_dim/2)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)  # (..., head_dim)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs  = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)                # (max_seq_len, head_dim/2)
        angles = torch.repeat_interleave(angles, 2, dim=-1)                           # (max_seq_len, head_dim)
        self.register_buffer('cos_cache', angles.cos())  # (max_seq_len, head_dim)
        self.register_buffer('sin_cache', angles.sin())  # (max_seq_len, head_dim)

    def forward(self, seq_len: int):
        # Returns (1, 1, seq_len, head_dim) — broadcasts over (B, n_heads, CL, head_dim)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, CL, head_dim)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, CL, head_dim)
        return cos, sin


# =============================================================================
# MoE components  (from moe_model.py)
# =============================================================================

class Expert(nn.Module):
    """SwiGLU expert: gate and value are separate projections, gate controls what passes."""

    def __init__(self, config: GQAMoEConfig) -> None:
        super().__init__()
        # hidden = (8/3) * ED, rounded to nearest multiple of 64 for hardware alignment
        hidden = int(2 * config.emb_dim * config.ffn_mult / 3)
        hidden = ((hidden + 63) // 64) * 64

        self.w_gate  = nn.Linear(config.emb_dim, hidden)   # (ED) → (hidden)  gate branch
        self.w_value = nn.Linear(config.emb_dim, hidden)   # (ED) → (hidden)  value branch
        self.w_proj  = nn.Linear(hidden, config.emb_dim)   # (hidden) → (ED)  output projection
        self.drop    = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_i, ED)  — only tokens dispatched to this expert
        gate  = F.silu(self.w_gate(x))   # (n_i, hidden)  — learned gate with SiLU activation
        value = self.w_value(x)          # (n_i, hidden)  — content to be gated
        hidden = gate * value            # (n_i, hidden)  — element-wise gating
        return self.drop(self.w_proj(hidden))  # (n_i, ED)


class Router(nn.Module):
    def __init__(self, config: GQAMoEConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k     = config.top_k
        self.gate = nn.Linear(config.emb_dim, config.n_experts, bias=False)  # (ED) → (N_experts)

    def forward(self, x: torch.Tensor):
        # x: (T, ED)
        T, ED = x.shape

        logits = self.gate(x)                # (T, N_experts)
        probs  = F.softmax(logits, dim=-1)   # (T, N_experts)

        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        # top_k_probs: (T, k)    top_k_idx: (T, k)

        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # (T, k)

        # Aux loss
        dispatch = torch.zeros(T, self.n_experts, device=x.device)  # (T, N_experts)
        dispatch.scatter_(dim=1, index=top_k_idx, value=1.0)
        f = dispatch.sum(dim=0) / (T * self.top_k)   # (N_experts,)
        p = probs.mean(dim=0)                          # (N_experts,)
        aux_loss = self.n_experts * (f * p).sum()      # scalar

        return top_k_weights, top_k_idx, aux_loss
        # top_k_weights: (T, k),  top_k_idx: (T, k),  aux_loss: scalar


class MoELayer(nn.Module):
    def __init__(self, config: GQAMoEConfig) -> None:
        super().__init__()
        self.router  = Router(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])

    def forward(self, x: torch.Tensor):
        # x: (B, CL, ED)
        B, CL, ED = x.shape
        T = B * CL

        x_flat = x.view(T, ED)                                          # (T, ED)
        top_k_weights, top_k_idx, aux_loss = self.router(x_flat)        # (T,k), (T,k), scalar
        output = torch.zeros_like(x_flat)                                # (T, ED)

        for i, expert in enumerate(self.experts):
            token_mask_2d = (top_k_idx == i)          # (T, k)  bool
            routed        = token_mask_2d.any(dim=-1)  # (T,)   bool
            if not routed.any():
                continue
            expert_out = expert(x_flat[routed])                           # (n_i, ED)
            weights_i  = top_k_weights[token_mask_2d].unsqueeze(-1)       # (n_i, 1)
            output[routed] += weights_i * expert_out                      # (n_i, ED)

        return output.view(B, CL, ED), aux_loss   # (B, CL, ED), scalar


# =============================================================================
# Grouped Query Attention
# =============================================================================
#
# Standard MHA:
#   Q, K, V each have n_heads independent projections.
#   Q: (ED) → (n_heads * head_dim)     K: (ED) → (n_heads * head_dim)
#
# GQA:
#   Q keeps all n_heads, but K and V only have n_kv_heads projections.
#   Q: (ED) → (n_heads * head_dim)     K: (ED) → (n_kv_heads * head_dim)
#
#   Before attention, K/V heads are repeated so each Q head has a K/V to pair with:
#   K: (B, n_kv_heads, CL, head_dim) → repeat → (B, n_heads, CL, head_dim)
#
# Special cases:
#   n_kv_heads == n_heads   →  standard MHA  (no sharing, 1:1)
#   n_kv_heads == 1         →  Multi-Query Attention  (all Q heads share 1 KV)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: GQAMoEConfig) -> None:
        super().__init__()
        self.n_heads    = config.n_heads       # total query heads
        self.n_kv_heads = config.n_kv_heads    # KV heads (fewer, shared across groups)
        self.head_dim   = config.emb_dim // config.n_heads
        self.n_rep      = config.n_heads // config.n_kv_heads  # Q heads per KV group

        # Q gets full projection — each query head is independent
        # W_q weight: (n_heads * head_dim, ED)
        self.W_q = nn.Linear(config.emb_dim, config.n_heads * self.head_dim)
        #   input:  (B, CL, ED)
        #   output: (B, CL, n_heads * head_dim) = (B, CL, ED)

        # K and V get smaller projections — only n_kv_heads worth
        # W_k weight: (n_kv_heads * head_dim, ED)
        self.W_k = nn.Linear(config.emb_dim, config.n_kv_heads * self.head_dim)
        #   input:  (B, CL, ED)
        #   output: (B, CL, n_kv_heads * head_dim)
        #   e.g. with n_kv_heads=2, head_dim=48:  (B, CL, 96)  vs full (B, CL, 384)

        # W_v weight: (n_kv_heads * head_dim, ED)
        self.W_v = nn.Linear(config.emb_dim, config.n_kv_heads * self.head_dim)
        #   same shape as W_k

        # Output projection — full size since we recombine all n_heads
        # W_o weight: (ED, n_heads * head_dim)
        self.W_o = nn.Linear(config.n_heads * self.head_dim, config.emb_dim)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x:   (B, CL, ED)
        # cos: (1, 1, CL, head_dim)
        # sin: (1, 1, CL, head_dim)
        B, CL, ED = x.shape

        # ── Project Q, K, V ──────────────────────────────────────────────────
        q = self.W_q(x)   # (B, CL, n_heads * head_dim)      e.g. (B, CL, 384)
        k = self.W_k(x)   # (B, CL, n_kv_heads * head_dim)   e.g. (B, CL, 96)
        v = self.W_v(x)   # (B, CL, n_kv_heads * head_dim)   e.g. (B, CL, 96)

        # ── Reshape into heads ────────────────────────────────────────────────
        q = q.view(B, CL, self.n_heads, self.head_dim)        # (B, CL, n_heads, head_dim)
        k = k.view(B, CL, self.n_kv_heads, self.head_dim)     # (B, CL, n_kv_heads, head_dim)
        v = v.view(B, CL, self.n_kv_heads, self.head_dim)     # (B, CL, n_kv_heads, head_dim)

        q = q.transpose(1, 2)   # (B, n_heads, CL, head_dim)      e.g. (B, 8, CL, 48)
        k = k.transpose(1, 2)   # (B, n_kv_heads, CL, head_dim)   e.g. (B, 2, CL, 48)
        v = v.transpose(1, 2)   # (B, n_kv_heads, CL, head_dim)   e.g. (B, 2, CL, 48)

        # ── Apply RoPE to Q and K ────────────────────────────────────────────
        # cos/sin are (1, 1, CL, head_dim) — broadcast over any number of heads
        # So RoPE applies identically whether we have n_heads or n_kv_heads
        q = q * cos + rotate_half(q) * sin   # (B, n_heads, CL, head_dim)
        k = k * cos + rotate_half(k) * sin   # (B, n_kv_heads, CL, head_dim)

        # ── Expand K, V to match Q's head count ──────────────────────────────
        # This is the core GQA operation: repeat each KV head n_rep times
        # so that every Q head has a K/V partner.
        #
        # With n_heads=8, n_kv_heads=2, n_rep=4:
        #   K before: (B, 2, CL, head_dim)   heads: [K0, K1]
        #   K after:  (B, 8, CL, head_dim)   heads: [K0, K0, K0, K0, K1, K1, K1, K1]
        #
        # Q0,Q1,Q2,Q3 all attend against K0  (group 0)
        # Q4,Q5,Q6,Q7 all attend against K1  (group 1)
        #
        # repeat_interleave repeats along dim=1, keeping heads grouped together
        k = torch.repeat_interleave(k, self.n_rep, dim=1)   # (B, n_heads, CL, head_dim)
        v = torch.repeat_interleave(v, self.n_rep, dim=1)   # (B, n_heads, CL, head_dim)

        # ── Attention ─────────────────────────────────────────────────────────
        # From here on, shapes are identical to standard MHA
        # Q, K, V: all (B, n_heads, CL, head_dim)
        dropout_p = self.resid_dropout.p if self.training else 0.0
        attn_w = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        # attn_w: (B, n_heads, CL, head_dim)

        # ── Recombine heads ───────────────────────────────────────────────────
        attn_w = attn_w.transpose(1, 2)    # (B, CL, n_heads, head_dim)
        attn_w = attn_w.reshape(B, CL, ED) # (B, CL, ED)

        # ── Output projection + residual dropout ─────────────────────────────
        out = self.resid_dropout(self.W_o(attn_w))  # (B, CL, ED)
        return out


# =============================================================================
# Transformer Block  (GQA attention + MoE FFN)
# =============================================================================

class GQAMoETransformerBlock(nn.Module):
    def __init__(self, config: GQAMoEConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.emb_dim)
        self.attn    = GroupedQueryAttention(config)
        self.ln_moe  = nn.LayerNorm(config.emb_dim)
        self.moe     = MoELayer(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: (B, CL, ED)

        # Attention sub-layer with pre-norm residual
        x = x + self.attn(self.ln_attn(x), cos, sin)       # (B, CL, ED)

        # MoE sub-layer with pre-norm residual — returns aux_loss
        moe_out, aux_loss = self.moe(self.ln_moe(x))        # (B, CL, ED), scalar
        x = x + moe_out                                      # (B, CL, ED)

        return x, aux_loss


# =============================================================================
# Full Model  (GQA + RoPE + MoE)
# =============================================================================

class GPTModelGQAMoE(nn.Module):
    def __init__(self, config: GQAMoEConfig):
        super().__init__()
        self.config  = config
        head_dim     = config.emb_dim // config.n_heads

        self.layer_dict = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.emb_dim),     # (vocab_size, ED)
            # No wpe — position encoded via RoPE
            drop = nn.Dropout(config.dropout),
            transformer_blocks = nn.ModuleList([
                GQAMoETransformerBlock(config)
                for _ in range(config.n_transformer_blocks)
            ]),
            ln_final = nn.LayerNorm(config.emb_dim)
        ))

        self.rope = RoPE(head_dim, config.context_len)

        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)     # (ED) → (vocab_size)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            # Scaled init for residual projections
            if pn.endswith('W_o.weight') or pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_transformer_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inpTox: torch.Tensor, targetTox: torch.Tensor):
        # inpTox:    (B, CL)
        # targetTox: (B, CL)  or None
        B, CL = inpTox.shape

        x = self.layer_dict.wte(inpTox)   # (B, CL, ED)
        x = self.layer_dict.drop(x)        # (B, CL, ED)

        cos, sin = self.rope(CL)           # (1, 1, CL, head_dim)

        # Accumulate aux loss from every MoE block
        total_aux_loss = torch.tensor(0.0, device=inpTox.device)
        for block in self.layer_dict.transformer_blocks:
            x, aux_loss = block(x, cos, sin)                  # (B, CL, ED), scalar
            total_aux_loss = total_aux_loss + aux_loss

        x = self.layer_dict.ln_final(x)                       # (B, CL, ED)

        if targetTox is not None:
            logits  = self.lm_head(x)                          # (B, CL, vocab_size)
            B, CL, V = logits.shape
            ce_loss = F.cross_entropy(logits.view(B * CL, V), targetTox.view(B * CL))
            mean_aux = total_aux_loss / self.config.n_transformer_blocks
            loss = ce_loss + self.config.aux_loss_coeff * mean_aux
        else:
            logits = self.lm_head(x[:, [-1], :])               # (B, 1, vocab_size)
            loss   = None

        return logits, loss

    def generate(self, inpTox: torch.Tensor, max_len: int, temperature=1.0, eot_token=None):
        for _ in range(max_len):
            inpCtx     = inpTox[:, -self.config.context_len:]
            logits, _  = self(inpCtx, None)
            logits     = logits[:, -1, :] / temperature        # (B, vocab_size)
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            if eot_token is not None and (next_token == eot_token).any():
                break
            inpTox = torch.cat([inpTox, next_token], dim=1)
        return inpTox


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    cfg = GQAMoEConfig(
        vocab_size=65, context_len=64, emb_dim=384,
        n_heads=8, n_kv_heads=2,
        n_transformer_blocks=4,
        n_experts=8, top_k=2
    )

    head_dim = cfg.emb_dim // cfg.n_heads
    n_rep    = cfg.n_heads // cfg.n_kv_heads

    print(f"=== GQA + RoPE + MoE Config ===")
    print(f"  emb_dim={cfg.emb_dim}, head_dim={head_dim}")
    print(f"  Q heads={cfg.n_heads}, KV heads={cfg.n_kv_heads}, Q heads per KV group={n_rep}")
    print(f"  n_experts={cfg.n_experts}, top_k={cfg.top_k}")
    print()

    # ── Parameter comparison: GQA vs MHA ──────────────────────────────────────
    # Q projection is the same: ED × (n_heads × head_dim) = 384 × 384
    # K/V projections shrink:
    #   MHA:  ED × (n_heads × head_dim)    = 384 × 384 = 147,456 each
    #   GQA:  ED × (n_kv_heads × head_dim) = 384 × 96  =  36,864 each
    #
    # Savings per block in K+V weights:
    #   MHA K+V:  2 × 147,456 = 294,912
    #   GQA K+V:  2 ×  36,864 =  73,728
    #   Saved:    221,184 params per block  (75% reduction in KV projection params)
    q_params  = cfg.emb_dim * (cfg.n_heads * head_dim)
    kv_params_mha = 2 * cfg.emb_dim * (cfg.n_heads * head_dim)
    kv_params_gqa = 2 * cfg.emb_dim * (cfg.n_kv_heads * head_dim)
    print(f"  Q projection params:       {q_params:,}")
    print(f"  KV projection params (MHA): {kv_params_mha:,}")
    print(f"  KV projection params (GQA): {kv_params_gqa:,}")
    print(f"  KV param reduction:          {(1 - kv_params_gqa/kv_params_mha)*100:.0f}%")
    print()

    # ── KV cache comparison at inference ──────────────────────────────────────
    # KV cache per token = 2 × n_kv_heads × head_dim × bytes_per_param
    # For CL=64, fp32:
    seq_len = 64
    kv_cache_mha = 2 * cfg.n_heads * seq_len * head_dim * 4     # fp32 bytes
    kv_cache_gqa = 2 * cfg.n_kv_heads * seq_len * head_dim * 4
    print(f"  KV cache for {seq_len} tokens (MHA): {kv_cache_mha:,} bytes")
    print(f"  KV cache for {seq_len} tokens (GQA): {kv_cache_gqa:,} bytes")
    print(f"  KV cache reduction:                   {(1 - kv_cache_gqa/kv_cache_mha)*100:.0f}%")
    print()

    # ── Forward pass ──────────────────────────────────────────────────────────
    B, CL = 2, 64
    x       = torch.randint(0, cfg.vocab_size, (B, CL))
    targets = torch.randint(0, cfg.vocab_size, (B, CL))

    model    = GPTModelGQAMoE(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total model params: {n_params:,}")
    print()

    logits, loss = model(x, targets)
    print(f"  train logits:     {tuple(logits.shape)}")     # (2, 64, 65)
    print(f"  train loss:       {loss.item():.4f}")

    logits_inf, loss_inf = model(x, None)
    print(f"  inference logits: {tuple(logits_inf.shape)}")  # (2, 1, 65)
    print(f"  inference loss:   {loss_inf}")                 # None
