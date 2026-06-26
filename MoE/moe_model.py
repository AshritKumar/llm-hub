"""
Mixture of Experts — Full Model (Steps 3 → 5)

    Step 3: MoELayer            — dispatches tokens to experts, collects weighted outputs
    Step 4: MoETransformerBlock — Attention + MoELayer (replaces dense FFN block)
    Step 5: GPTModelMoE         — full GPT with RoPE + MoE, accumulates aux loss

Builds on concepts from:
    step1_expert_router.py  (Expert, Router)
    gpt_rope_model.py       (RoPE, rotate_half, MultiHeadAttention)
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
class MoEConfig:
    # ── Standard GPT fields ──────────────────────────────────────────────────
    vocab_size:          int   = 65
    context_len:         int   = 256
    emb_dim:             int   = 384
    n_heads:             int   = 6
    n_transformer_blocks:int   = 6
    dropout:             float = 0.05
    # ── MoE-specific fields ──────────────────────────────────────────────────
    n_experts:           int   = 8     # total expert FFNs per MoE block
    top_k:               int   = 2     # experts each token is dispatched to
    ffn_mult:            int   = 4     # expert hidden dim = emb_dim * ffn_mult
    aux_loss_coeff:      float = 0.01  # α  — scales aux loss vs. CE loss


# =============================================================================
# RoPE helpers  (from gpt_rope_model.py)
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., head_dim) — interleaved pairs (q1, q2, q3, q4, ...)
    x1 = x[..., ::2]    # (..., head_dim/2) — first element of each pair
    x2 = x[..., 1::2]   # (..., head_dim/2) — second element of each pair
    # stack → (..., head_dim/2, 2) then flatten last two → (..., head_dim)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs  = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
        angles = torch.repeat_interleave(angles, 2, dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer('cos_cache', angles.cos())
        self.register_buffer('sin_cache', angles.sin())

    def forward(self, seq_len: int):
        # Returns [1, 1, seq_len, head_dim] — broadcasts over (B, n_head, CL, head_dim)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin


# =============================================================================
# Step 1 recap — Expert and Router  (full detail in step1_expert_router.py)
# =============================================================================

class Expert(nn.Module):
    """Single FFN. Receives only the tokens dispatched to it, not the full batch."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        hidden = config.emb_dim * config.ffn_mult
        self.fc   = nn.Linear(config.emb_dim, hidden)   # ED → ED*4
        self.gelu = nn.GELU()
        self.proj = nn.Linear(hidden, config.emb_dim)   # ED*4 → ED
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   (n_i, ED)  — only tokens dispatched to this expert
        # out: (n_i, ED)
        return self.drop(self.proj(self.gelu(self.fc(x))))


class Router(nn.Module):
    """Top-k router with load-balancing auxiliary loss."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k     = config.top_k
        self.gate = nn.Linear(config.emb_dim, config.n_experts, bias=False)
        # gate weight: (N_experts, ED)

    def forward(self, x: torch.Tensor):
        # x: (T, ED)  where T = B * CL
        T, ED = x.shape

        logits = self.gate(x)               # (T, N_experts)
        probs  = F.softmax(logits, dim=-1)  # (T, N_experts)

        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        # top_k_probs: (T, k)   top_k_idx: (T, k)

        # Renormalize k weights to sum to 1
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        # top_k_weights: (T, k)

        # Aux loss — see step1_expert_router.py for full explanation
        dispatch = torch.zeros(T, self.n_experts, device=x.device)  # (T, N_experts)
        dispatch.scatter_(dim=1, index=top_k_idx, value=1.0)

        f = dispatch.sum(dim=0) / (T * self.top_k)  # (N_experts,)  fraction dispatched
        p = probs.mean(dim=0)                         # (N_experts,)  mean router prob

        aux_loss = self.n_experts * (f * p).sum()     # scalar

        return top_k_weights, top_k_idx, aux_loss


# =============================================================================
# Step 3 — MoELayer
# =============================================================================
#
# Combines N experts + router into a single drop-in replacement for a dense FFN.
#
# Forward logic:
#   1. Flatten (B, CL, ED) → (T, ED)
#   2. Router assigns each token to k experts with weights
#   3. Dispatch: for each expert, gather its tokens, run them through, scatter back
#   4. Weighted-sum expert contributions for each token
#   5. Unflatten (T, ED) → (B, CL, ED)
#   6. Return (output, aux_loss)

class MoELayer(nn.Module):
    """
    MoE replacement for a dense FFN.
    Each token is processed by top-k of N experts; outputs are weighted-summed.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.router  = Router(config)
        # N independent experts — same architecture, completely separate weights
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])

    def forward(self, x: torch.Tensor):
        # x: (B, CL, ED)
        B, CL, ED = x.shape
        T = B * CL

        # ── Flatten to treat every token independently ────────────────────────
        x_flat = x.view(T, ED)                  # (T, ED)

        # ── Route ─────────────────────────────────────────────────────────────
        top_k_weights, top_k_idx, aux_loss = self.router(x_flat)
        # top_k_weights: (T, k)   top_k_idx: (T, k)

        # ── Dispatch → Expert → Collect ───────────────────────────────────────
        output = torch.zeros_like(x_flat)       # (T, ED) accumulation buffer

        for i, expert in enumerate(self.experts):

            # Which tokens are routed to expert i?
            # token_mask_2d[t, ki] = True  if top_k_idx[t, ki] == i
            token_mask_2d = (top_k_idx == i)     # (T, k)  bool
            routed        = token_mask_2d.any(dim=-1)  # (T,)  bool — dispatched tokens

            if not routed.any():
                continue                         # expert received no tokens this step

            # Run only the dispatched tokens through this expert
            expert_out = expert(x_flat[routed])  # (n_i, ED)

            # Extract the weight assigned to expert i for each dispatched token.
            # token_mask_2d[routed] is (n_i, k) with exactly one True per row
            # (a token appears at most once in top-k for a given expert).
            # Boolean indexing on (T, k) with (T, k) mask scans row-by-row,
            # so the n_i values come out in the same order as x_flat[routed].
            weights_i = top_k_weights[token_mask_2d].unsqueeze(-1)  # (n_i, 1)

            # Weighted add back to the output positions these tokens came from
            output[routed] += weights_i * expert_out   # (n_i, ED) broadcast + scatter

        # ── Restore original shape ─────────────────────────────────────────────
        return output.view(B, CL, ED), aux_loss  # (B, CL, ED),  scalar


# =============================================================================
# Step 4 — MoETransformerBlock
# =============================================================================
#
# Identical to a standard TransformerBlock except FFN → MoELayer.
# Propagates aux_loss upward so the model can accumulate it.

class MultiHeadAttention(nn.Module):
    """Causal MHA with RoPE. Identical to gpt_rope_model.py."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv    = nn.Linear(config.emb_dim, 3 * config.emb_dim)
        self.w0     = nn.Linear(config.emb_dim, config.emb_dim)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: (B, CL, ED)
        B, CL, ED = x.shape
        head_dim   = ED // self.config.n_heads

        q, k, v = self.qkv(x).split(ED, dim=2)
        # Each: (B, CL, ED)

        q = q.view(B, CL, self.config.n_heads, head_dim).transpose(1, 2)
        k = k.view(B, CL, self.config.n_heads, head_dim).transpose(1, 2)
        v = v.view(B, CL, self.config.n_heads, head_dim).transpose(1, 2)
        # Each: (B, n_head, CL, head_dim)

        # Apply RoPE to Q and K — V carries content, not position
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        dropout_p = self.config.dropout if self.training else 0.0
        attn_w = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        # attn_w: (B, n_head, CL, head_dim)

        attn_w = attn_w.transpose(1, 2).reshape(B, CL, ED)  # (B, CL, ED)
        return self.resid_dropout(self.w0(attn_w))           # (B, CL, ED)


class MoETransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with MoE in place of a dense FFN.
    Returns (output, aux_loss) so the model can accumulate aux losses.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.emb_dim)
        self.attn    = MultiHeadAttention(config)
        self.ln_moe  = nn.LayerNorm(config.emb_dim)
        self.moe     = MoELayer(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: (B, CL, ED)

        # Attention sub-layer — standard residual, no aux loss
        x = x + self.attn(self.ln_attn(x), cos, sin)       # (B, CL, ED)

        # MoE sub-layer — residual + collect aux loss
        moe_out, aux_loss = self.moe(self.ln_moe(x))        # (B, CL, ED),  scalar
        x = x + moe_out                                      # (B, CL, ED)

        return x, aux_loss


# =============================================================================
# Step 5 — GPTModelMoE
# =============================================================================
#
# Full GPT model:
#   token embedding (no positional embedding — position handled by RoPE)
#   → N MoETransformerBlocks (each returns its own aux_loss)
#   → final LayerNorm
#   → lm_head (weight-tied to wte)
#
# Loss = CE_loss + aux_loss_coeff * mean(aux_loss over all blocks)

class GPTModelMoE(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config   = config
        head_dim      = config.emb_dim // config.n_heads

        self.layer_dict = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.emb_dim),
            # No wpe — position encoded via RoPE inside each attention layer
            drop = nn.Dropout(config.dropout),
            transformer_blocks = nn.ModuleList([
                MoETransformerBlock(config)
                for _ in range(config.n_transformer_blocks)
            ]),
            ln_final = nn.LayerNorm(config.emb_dim)
        ))

        # Shared RoPE cache — same angles applied across all layers and heads
        self.rope = RoPE(head_dim, config.context_len)

        # Unembedding — weight-tied to wte so they share the same matrix
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)
        self.lm_head.weight = nn.Parameter(self.layer_dict.wte.weight)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w0.weight') or pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_transformer_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inpTox: torch.Tensor, targetTox: torch.Tensor):
        # inpTox, targetTox: (B, CL)
        B, CL = inpTox.shape

        x = self.layer_dict.wte(inpTox)  # (B, CL, ED) — no positional embedding added
        x = self.layer_dict.drop(x)

        cos, sin = self.rope(CL)         # (1, 1, CL, head_dim)

        # ── Transformer blocks — accumulate aux loss from each MoE block ──────
        total_aux_loss = torch.tensor(0.0, device=inpTox.device)
        for block in self.layer_dict.transformer_blocks:
            x, aux_loss = block(x, cos, sin)     # (B, CL, ED),  scalar
            total_aux_loss = total_aux_loss + aux_loss

        x = self.layer_dict.ln_final(x)          # (B, CL, ED)

        # ── Loss computation ──────────────────────────────────────────────────
        if targetTox is not None:
            logits = self.lm_head(x)             # (B, CL, vocab_size)
            B, CL, V = logits.shape

            ce_loss  = F.cross_entropy(logits.view(B * CL, V), targetTox.view(B * CL))

            # Average aux loss across blocks so aux_loss_coeff is independent of depth
            mean_aux = total_aux_loss / self.config.n_transformer_blocks

            loss = ce_loss + self.config.aux_loss_coeff * mean_aux
        else:
            # Inference: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss   = None

        return logits, loss

    def generate(self, inpTox: torch.Tensor, max_len: int, temperature=1.0, eot_token=None):
        for _ in range(max_len):
            inpCtx     = inpTox[:, -self.config.context_len:]
            logits, _  = self(inpCtx, None)
            logits     = logits[:, -1, :] / temperature   # (B, vocab_size)
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
    cfg = MoEConfig(
        vocab_size=65, context_len=64, emb_dim=384,
        n_heads=6, n_transformer_blocks=4,
        n_experts=8, top_k=2
    )

    B, CL = 2, 64
    x       = torch.randint(0, cfg.vocab_size, (B, CL))
    targets = torch.randint(0, cfg.vocab_size, (B, CL))

    model  = GPTModelMoE(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    logits, loss = model(x, targets)
    print(f"logits:  {tuple(logits.shape)}")   # (2, 64, 65)
    print(f"loss:    {loss.item():.4f}")        # CE + aux, should be ~ln(65) ≈ 4.17

    # Check inference path returns (B, 1, V) not (B, V)
    logits_inf, loss_inf = model(x, None)
    print(f"inference logits: {tuple(logits_inf.shape)}")  # (2, 1, 65)
    print(f"inference loss:   {loss_inf}")                 # None
