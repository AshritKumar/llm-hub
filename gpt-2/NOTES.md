# GPT-2 Replica Notes

## nn.Linear weight shape: (out, in)

`nn.Linear(in, out)` is declared as `(in, out)` but stores weights as `(out, in)` and computes `x @ W.T`.

**Why:** storing `(out, in)` means each output neuron's weights occupy a contiguous row in memory (C-order). During the matmul, reading a row is cache-friendly. If stored as `(in, out)`, each neuron's weights would be a column — strided, non-contiguous, cache-unfriendly.

HF's `Conv1D` is the opposite — stored as `(in, out)`, computes `x @ W` directly (no transpose). So when copying HF GPT-2 weights into `nn.Linear`, these layers need `.t()`:

```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
```

Only these four per block need transposing — everything else (`Embedding`, `LayerNorm`) is not a `Conv1D`.

## L2 Norm and Gradient Clipping

### L2 Norm

The L2 norm (Euclidean norm) of a vector measures its "length":

```
||v||₂ = sqrt(v₁² + v₂² + v₃² + ... + vₙ²)
```

For a vector `[3, 4]`, the L2 norm is `sqrt(9 + 16) = 5`.

### L2 Norm of Gradients

A model has millions of parameters, each with its own gradient. `clip_grad_norm_` treats all gradients across all parameters as a single flat vector and computes its L2 norm:

```
||grad||₂ = sqrt(sum of (grad_i² for every parameter i in every layer))
```

This is the *global* gradient norm — a single scalar representing how large the overall update is.

### Gradient Clipping

If `||grad||₂ > max_norm` (typically 1.0), all gradients are rescaled proportionally:

```
grad = grad * (max_norm / ||grad||₂)
```

This preserves the direction of the update but limits its magnitude. After clipping, `||grad||₂ == max_norm` exactly.

### Why It's Needed

During early training, loss landscapes can be steep and gradients can spike to very large values (gradient explosion), causing parameter updates that overshoot and destabilize training. Clipping prevents this.

**Important:** clipping only handles large-but-finite gradients. If gradients are `NaN` (e.g. from numerical overflow in a fused op under mixed precision), `NaN * scalar = NaN` — clipping cannot recover from that.

## Variance Accumulation and the sqrt(N) Rule

The core rule: **Var(X + Y) = Var(X) + Var(Y)** for independent variables. Variance adds linearly; std grows as sqrt(N).

### Residual stream accumulation

Each transformer block adds two c_proj outputs to the residual stream (one from attn, one from MLP) — 24 additions total for 12 layers. If each has std=0.02 (var=0.0004):

```
total variance = 24 * 0.0004 = 0.0096
total std      = sqrt(0.0096) = sqrt(24) * 0.02 ≈ 0.098
```

**Fix:** init c_proj with `std = 0.02 / sqrt(2 * n_layers)` so each contribution's variance is `0.0004 / 24`. After 24 additions the total std is back to 0.02 — the sqrt(24) cancels out.

### Attention dot product scaling

`q @ k.T` is a sum of `head_dim` element-wise products. If q and k have mean=0 and variance=1:

```
Var(qᵢ * kᵢ) = Var(qᵢ) * Var(kᵢ) = 1
Var(q @ k.T) = head_dim * 1 = head_dim
Std(q @ k.T) = sqrt(head_dim)
```

Large std pushes softmax toward near-one-hot outputs (near-zero gradients). **Fix:** multiply by `head_dim ** -0.5` = `1 / sqrt(head_dim)` at every forward pass to bring variance back to 1.

### The pattern

Both fixes are inverses of the same rule:
- Residual: scale each *input* down by `sqrt(N)` at init so the accumulated output is well-scaled
- Attention: scale the *output* down by `sqrt(head_dim)` at runtime after the summation happens
