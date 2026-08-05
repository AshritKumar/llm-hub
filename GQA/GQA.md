# Grouped Query Attention (GQA)

## The problem: KV cache is the inference bottleneck

In standard Multi-Head Attention (MHA), every head has its **own Q, K, and V** projections.
During autoregressive inference we cache K and V for all past tokens so we don't recompute
them. This KV cache grows as:

```
KV cache = 2 × n_heads × seq_len × head_dim × bytes_per_param
```

For a model like LLaMA-70B (64 heads, 8192 context, head_dim=128, fp16):
```
= 2 × 64 × 8192 × 128 × 2 bytes  ≈  4 GB per sequence
```

At batch size 32 that's 128 GB just for KV cache — often more than the model weights.
The KV cache, not the model, becomes the memory bottleneck during inference.

---

## The insight

Q encodes *"what am I looking for?"* — each head needs its own nuanced query.
K encodes *"what do I contain?"* — this is more uniform across heads.
V encodes *"what value to return if matched"* — similarly uniform.

Q needs many independent heads for expressiveness. K and V can be **shared across
groups of Q heads** with minimal quality loss.

---

## The three attention variants

```
Multi-Head Attention  (MHA):    Q heads = 8,   KV heads = 8    (1:1, no sharing)
Grouped Query Attn    (GQA):    Q heads = 8,   KV heads = 2    (4:1, grouped)
Multi-Query Attn      (MQA):    Q heads = 8,   KV heads = 1    (8:1, all share one)
```

GQA is a **generalisation** — MHA and MQA are its two extremes.

Visually with n_heads=8, n_kv_heads=2:

```
Q heads:    Q0  Q1  Q2  Q3  |  Q4  Q5  Q6  Q7
               ↓                    ↓
KV heads:     K0, V0              K1, V1

Group 0: Q0, Q1, Q2, Q3  all attend using K0, V0
Group 1: Q4, Q5, Q6, Q7  all attend using K1, V1
```

---

## What changes from standard MHA

Only two things change. Everything else is identical.

### 1. K and V projections shrink

```
MHA:
    W_q: (ED, n_heads * head_dim)       = (384, 384)   — full
    W_k: (ED, n_heads * head_dim)       = (384, 384)   — full
    W_v: (ED, n_heads * head_dim)       = (384, 384)   — full

GQA (n_kv_heads=2):
    W_q: (ED, n_heads * head_dim)       = (384, 384)   — same, unchanged
    W_k: (ED, n_kv_heads * head_dim)    = (384, 96)    — 4× smaller
    W_v: (ED, n_kv_heads * head_dim)    = (384, 96)    — 4× smaller
```

After projection:
```
q: (B, CL, n_heads * head_dim)      = (B, CL, 384)   — 8 independent heads
k: (B, CL, n_kv_heads * head_dim)   = (B, CL, 96)    — only 2 heads
v: (B, CL, n_kv_heads * head_dim)   = (B, CL, 96)    — only 2 heads
```

### 2. K and V are expanded before attention

After reshaping into heads:
```
q: (B, n_heads, CL, head_dim)      = (B, 8, CL, 48)
k: (B, n_kv_heads, CL, head_dim)   = (B, 2, CL, 48)
v: (B, n_kv_heads, CL, head_dim)   = (B, 2, CL, 48)
```

Q has 8 heads but K/V have 2. We can't compute `Q @ K^T` with mismatched head dims.
So we repeat each KV head `n_rep = n_heads // n_kv_heads = 4` times:

```python
k = torch.repeat_interleave(k, n_rep, dim=1)
v = torch.repeat_interleave(v, n_rep, dim=1)
```

```
k before repeat: (B, 2, CL, 48)   heads: [K0, K1]
k after repeat:  (B, 8, CL, 48)   heads: [K0, K0, K0, K0, K1, K1, K1, K1]
```

Now Q, K, V are all `(B, n_heads, CL, head_dim)` and attention proceeds as standard MHA.

**Important:** this repeat is a **view-level operation** in practice — it doesn't
allocate 4× the memory for K/V. The underlying tensor data is shared. The cost is in
the attention computation (which scales with n_heads regardless), not in storing
duplicated KV data.

---

## RoPE with GQA

RoPE applies to Q and K independently based on position and head_dim.
The cos/sin cache is `(1, 1, CL, head_dim)` — it broadcasts over **any number of heads**.

So for K with fewer heads, the exact same RoPE works:
```
q = q * cos + rotate_half(q) * sin    # (B, n_heads, CL, head_dim)
k = k * cos + rotate_half(k) * sin    # (B, n_kv_heads, CL, head_dim)
```

RoPE is applied **before** the repeat_interleave expansion. This is correct and
efficient — rotating 2 KV heads is cheaper than rotating 8.

---

## KV cache savings

The whole point of GQA is reducing KV cache at inference time.

```
KV cache per token:
    MHA:  2 × n_heads × head_dim      = 2 × 8 × 48 = 768 values
    GQA:  2 × n_kv_heads × head_dim   = 2 × 2 × 48 = 192 values

    Reduction: n_heads / n_kv_heads = 4×
```

From our sanity check (64 tokens, fp32):
```
MHA KV cache:  196,608 bytes
GQA KV cache:   49,152 bytes   (75% reduction)
```

This scales linearly with sequence length. At 8192 context the savings become significant.

---

## Projection parameter savings

```
KV projection params per block:
    MHA:  2 × ED × (n_heads × head_dim)      = 2 × 384 × 384 = 294,912
    GQA:  2 × ED × (n_kv_heads × head_dim)   = 2 × 384 × 96  =  73,728

    Reduction: 75%  (for n_heads=8, n_kv_heads=2)
```

Q projection and output projection are unchanged — only K/V shrink.

---

## Choosing n_kv_heads

| n_kv_heads | Name | KV cache | Quality |
|---|---|---|---|
| n_heads (8) | MHA | full | baseline |
| n_heads/2 (4) | GQA-4 | 2× smaller | ~baseline |
| n_heads/4 (2) | GQA-2 | 4× smaller | very close to baseline |
| 1 | MQA | n_heads× smaller | slight degradation |

Empirical findings (from the GQA paper):
- GQA with ~8 KV heads matches MHA quality across most benchmarks
- MQA (1 KV head) shows measurable quality loss on some tasks
- The sweet spot is typically n_kv_heads = n_heads / 4 to n_heads / 8

Models using GQA in production:
- LLaMA 2 70B: 64 Q heads, 8 KV heads  (8:1)
- LLaMA 3: all sizes use GQA
- Mistral 7B: 32 Q heads, 8 KV heads  (4:1)
- Gemma: uses GQA across all variants

---

## SwiGLU — Expert Activation Function

### Why not GELU?

In a GELU FFN, every hidden dimension is activated independently based only on its own value:

```
GELU FFN:
    hidden = GELU(Linear_1(x))   # (n, 4*ED) — each dim self-gated
    output = Linear_2(hidden)     # (n, ED)

    2 matrices, activation applied element-wise
```

There's no interaction between dimensions inside the activation step. A hidden dimension
with value `-0.5` always gets scaled to `~-0.15` regardless of the input.

### SwiGLU: learned gating

SwiGLU splits the computation into two parallel branches — a **gate branch** that learns
what to let through, and a **value branch** that carries the content:

```
SwiGLU FFN:
    gate   = SiLU(W_gate(x))     # (n, hidden) — controls what passes
    value  = W_value(x)           # (n, hidden) — the actual content
    hidden = gate * value         # (n, hidden) — element-wise gating
    output = W_proj(hidden)       # (n, ED)

    3 matrices, gate and value are different projections of the same input
```

`SiLU(x) = x * sigmoid(x)` (also called Swish).

The gate and value are two **different linear projections** of the same input. This lets
the network learn "let dimension i through when the input looks like X, suppress it when
it looks like Y" — a richer decision boundary than GELU's fixed per-element curve.

### FLOPs matching

A matmul of `(1, A) × (A, B)` costs `2 × A × B` FLOPs per token.

```
GELU FFN (hidden = 4*ED):
    Linear_1:  2 × ED × 4*ED  =  8*ED²
    Linear_2:  2 × 4*ED × ED  =  8*ED²
    Total:     16*ED²

SwiGLU FFN (hidden = h):
    W_gate:    2 × ED × h
    W_value:   2 × ED × h
    W_proj:    2 × h × ED
    Total:     6 × ED × h
```

Setting `6 × ED × h = 16 × ED²` gives `h = (8/3) × ED ≈ 2.67 × ED`.

With ED=384: `h = 1024` (vs GELU's `4 × 384 = 1536`).

3 smaller matrices instead of 2 larger ones — same FLOPs, same parameter count, but
consistently better training loss across all benchmarks. This is why every modern
architecture (LLaMA, Mistral, Gemma, DeepSeek) uses SwiGLU.

### Hidden dim alignment

The raw `(8/3) * ED` may not be hardware-friendly. We round up to the nearest multiple
of 64 for GPU memory alignment:

```python
hidden = int(2 * ED * ffn_mult / 3)       # raw: (8/3) * ED when ffn_mult=4
hidden = ((hidden + 63) // 64) * 64        # round UP to nearest 64
```

`+ 63` is `multiple - 1` — the exact offset that pushes non-aligned values to the
ceiling without overshooting already-aligned ones.

---

## Full architecture in gqa_model.py

The model combines four techniques:

```
GPTModelGQAMoE
  ├── Token embedding (no positional embedding)
  ├── RoPE (position injected into Q/K inside attention)
  ├── N × GQAMoETransformerBlock
  │     ├── LayerNorm → GroupedQueryAttention (GQA + RoPE) → residual
  │     └── LayerNorm → MoELayer (Router + N SwiGLU Experts) → residual + aux_loss
  ├── Final LayerNorm
  └── lm_head (separate unembedding, not weight-tied)

Loss = CE_loss + aux_loss_coeff × mean(aux_loss across blocks)
```
