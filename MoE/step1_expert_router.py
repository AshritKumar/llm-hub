"""
Mixture of Experts — Step 1 & 2: Expert and Router

Standard GPT block:   x → Attn → FFN → out
MoE block:            x → Attn → Router → top-k Experts (weighted sum) → out

This file implements the two core building blocks before we assemble the full model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MoEConfig:
    emb_dim:         int   = 384   # hidden / embedding dimension (ED)
    n_experts:       int   = 8     # total number of expert FFNs
    top_k:           int   = 2     # experts each token is routed to (k)
    dropout:         float = 0.1
    ffn_mult:        int   = 4     # expert hidden dim = emb_dim * ffn_mult
    aux_loss_coeff:  float = 0.01  # α that scales aux loss against CE loss at training time


# =============================================================================
# Step 1 — Expert
# =============================================================================
#
# An Expert is just a standard FFN.
# N experts = N *independent* FFNs with the same architecture but separate weights.
# Each expert learns to specialize in different token types / concepts over training.
#
# Crucially: a token is only *dispatched* to k of these experts, so the other
# (N - k) experts never see it and never receive a gradient for it this step.

class Expert(nn.Module):
    """
    Single FFN expert.  Architecture: Linear → GELU → Linear → Dropout.
    Input is only the subset of tokens routed here — not the full batch.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        hidden = config.emb_dim * config.ffn_mult
        self.fc   = nn.Linear(config.emb_dim, hidden)           # ED → ED*4
        self.gelu = nn.GELU()
        self.proj = nn.Linear(hidden, config.emb_dim)           # ED*4 → ED
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   (n_i, ED)   — only the n_i tokens dispatched to this expert
        # out: (n_i, ED)
        return self.drop(self.proj(self.gelu(self.fc(x))))


# =============================================================================
# Step 2 — Router
# =============================================================================
#
# The router answers: "for each token, which k experts should process it?"
#
# Three responsibilities per forward pass:
#   (a) Compute a probability over all N experts for every token
#   (b) Select top-k experts and produce their normalized dispatch weights
#   (c) Compute an auxiliary load-balancing loss to prevent routing collapse
#       (all tokens piling onto 1-2 experts while the rest starve)

class Router(nn.Module):
    """
    Top-k router with Switch-Transformer-style load-balancing auxiliary loss.

    Returns:
        top_k_weights : (T, k)   — renormalized dispatch weights summing to 1
        top_k_idx     : (T, k)   — which expert each weight corresponds to
        aux_loss      : scalar   — added to the main CE loss during training
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k     = config.top_k
        # One linear layer: hidden state → score for each expert
        # No bias — we only care about relative scores, not absolute magnitudes
        self.gate = nn.Linear(config.emb_dim, config.n_experts, bias=False)
        # gate weight: (N_experts, ED)

    def forward(self, x: torch.Tensor):
        # x: (T, ED)    T = B * CL — all tokens in the batch, flattened
        T, ED = x.shape

        # ── (a) Per-token probability over all experts ────────────────────────
        logits = self.gate(x)               # (T, N_experts)  raw scores
        probs  = F.softmax(logits, dim=-1)  # (T, N_experts)  each row sums to 1

        # ── (b) Top-k selection ───────────────────────────────────────────────
        # Pick the k experts with the highest probability for each token
        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        # top_k_probs: (T, k)  raw softmax weights for the k chosen experts
        # top_k_idx:   (T, k)  expert indices, one row per token

        # Renormalize so the k chosen weights sum to 1 (not the full N).
        # This keeps the weighted sum of expert outputs at unit scale regardless of k.
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        # top_k_weights: (T, k)  — final dispatch weights, Σ per row = 1

        # ── (c) Auxiliary load-balancing loss ─────────────────────────────────
        #
        # Without this, the router collapses: it finds 1-2 experts it likes early
        # in training and routes everything there, starving the rest permanently.
        #
        # We want to equalize two quantities across experts:
        #
        #   f_i  = fraction of tokens actually dispatched to expert i
        #          computed from the hard top-k decision → NOT differentiable
        #
        #   p_i  = mean router probability given to expert i across all tokens
        #          computed from softmax → differentiable
        #
        # aux_loss = N * Σ_i (f_i * p_i)
        #
        # Gradient flows only through p_i to the gate weights.
        # f_i scales the gradient: overloaded experts get a stronger push to equalize.
        # The N multiplier keeps the loss magnitude stable as N grows.

        # Build a binary dispatch mask — 1 where a token was routed to an expert
        # One row per token, exactly k ones per row
        dispatch = torch.zeros(T, self.n_experts, device=x.device)  # (T, N_experts)
        dispatch.scatter_(
            dim=1,
            index=top_k_idx,    # (T, k)  — column positions to fill with 1
            value=1.0
        )
        # dispatch: (T, N_experts)

        # f_i: fraction of total token-expert assignments going to expert i
        # Dividing by T*top_k normalises so that Σ_i f_i = 1
        f = dispatch.sum(dim=0) / (T * self.top_k)   # (N_experts,)

        # p_i: mean softmax probability assigned to expert i across all T tokens
        p = probs.mean(dim=0)                          # (N_experts,)

        aux_loss = self.n_experts * (f * p).sum()      # scalar

        return top_k_weights, top_k_idx, aux_loss
        # top_k_weights: (T, k)
        # top_k_idx:     (T, k)
        # aux_loss:      scalar


# =============================================================================
# Quick sanity check — run this file directly to verify shapes
# =============================================================================

if __name__ == "__main__":
    cfg = MoEConfig(emb_dim=384, n_experts=8, top_k=2)

    B, CL, ED = 4, 16, cfg.emb_dim
    T = B * CL                                          # 64 tokens

    x = torch.randn(T, ED)                              # (64, 384)

    # --- Router ---
    router = Router(cfg)
    weights, idx, aux = router(x)
    print(f"Router")
    print(f"  input:           {tuple(x.shape)}")       # (64, 384)
    print(f"  top_k_weights:   {tuple(weights.shape)}") # (64, 2)
    print(f"  top_k_idx:       {tuple(idx.shape)}")     # (64, 2)
    print(f"  aux_loss:        {aux.item():.4f}")        # scalar ~1/N_experts = 0.125

    # --- Expert (one of N, receives only its dispatched tokens) ---
    expert = Expert(cfg)
    n_dispatched = 20                                   # pretend 20 tokens routed here
    x_dispatched = torch.randn(n_dispatched, ED)        # (20, 384)
    out = expert(x_dispatched)
    print(f"\nExpert")
    print(f"  input:   {tuple(x_dispatched.shape)}")    # (20, 384)
    print(f"  output:  {tuple(out.shape)}")             # (20, 384)

    # --- Uniform routing check ---
    # With random weights, aux_loss ≈ 1.0 (N * 1/N * 1/N * N = 1)
    # After training converges with good balancing it should stay near 1.0
    print(f"\nAux loss reference: perfectly uniform = {1.0:.4f}, got {aux.item():.4f}")
