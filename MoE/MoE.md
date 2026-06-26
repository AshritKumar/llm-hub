# Mixture of Experts (MoE)

## Why MoE?

In a standard Transformer every token passes through the **same FFN** in every block.
This couples capacity (how much the model can know) directly to compute (how much it spends per token).

MoE breaks that coupling.

```
Standard block:   x → Attn → FFN (one, shared) → out
MoE block:        x → Attn → Router → Expert_k (sparse, per-token) → out
```

You can increase total parameters by adding more experts without increasing the compute per
token — only k of the N experts fire for any given token.

---

## Components

### Expert

An Expert is just a standard FFN — Linear → GELU → Linear → Dropout.
Nothing special about its architecture. What makes it an "expert" is that it has
**completely independent weights** from every other expert, and it only ever sees the
**subset of tokens routed to it**, not the full batch.

Over training, each expert's weights specialise for the kinds of tokens it keeps receiving.

```
Expert i receives:   (n_i, ED)   — only its dispatched tokens
Expert i outputs:    (n_i, ED)
```

n_i varies each step and is typically much smaller than T (total tokens in the batch).

---

### Router

The router answers: *"for this token's content, which k experts should process it?"*

```
gate = nn.Linear(ED, N_experts, bias=False)   # weight: (N_experts, ED)

logits        = gate(x)              # (T, N_experts)  — raw scores
probs         = softmax(logits)      # (T, N_experts)  — rows sum to 1
top_k_probs, top_k_idx = topk(probs, k)
top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
```

Each row of the gate weight `(1, ED)` is a learned "what does expert i prefer to see?" vector.
The dot product with a token's hidden state measures alignment — how well this token's content
matches what expert i has specialised in. Routing is purely content-driven; sequence position
plays no role.

The k chosen weights are **renormalised** to sum to 1 so the weighted sum of expert outputs
stays at unit scale regardless of k.

---

### Auxiliary Load-Balancing Loss

#### The problem it solves

Without any pressure to distribute tokens, the router quickly finds 1–2 experts it prefers
early in training and routes almost everything there. Those experts get abundant gradients and
improve rapidly; the rest are starved and never learn. This is called **routing collapse**.

The auxiliary loss adds a soft penalty that encourages the router to spread tokens evenly
across all experts.

---

#### Two quantities per expert

**`f_i` — fraction of tokens actually dispatched to expert i**

Built from the hard top-k decision: a binary dispatch matrix with exactly k ones per row
(one per chosen expert per token).

```python
dispatch = zeros(T, N_experts)
dispatch.scatter_(dim=1, index=top_k_idx, value=1.0)
# dispatch: (T, N_experts)  — each row has exactly k ones
```

Summing over tokens gives the raw count of assignments to each expert.
Dividing normalises into a fraction:

```python
f = dispatch.sum(dim=0) / (T * top_k)   # (N_experts,)
```

**Why divide by `T * top_k` and not just `T`?**

`dispatch.sum(dim=0).sum()` = `T * top_k` because there are exactly k ones per row across
T rows. Dividing by `T * top_k` makes `Σ_i f_i = 1` — a proper probability distribution.

If you divided by just `T`, then `Σ_i f_i = top_k`, and the loss magnitude would grow with k,
making the coefficient `aux_loss_coeff` harder to tune.

With perfectly balanced routing every expert gets `1/N` of the total slots:
```
f_i = (T * top_k / N) / (T * top_k)  =  1/N   ✓
```

**`f_i` is NOT differentiable** — it comes from a discrete argmax (top-k).
Gradients cannot flow through it.

---

**`p_i` — mean router probability assigned to expert i across all tokens**

```python
p = probs.mean(dim=0)    # (N_experts,)
```

`p_i` uses the full softmax output (not the hard top-k decision), so it **is differentiable**.
`Σ_i p_i = 1` always.

---

#### The loss formula

```python
aux_loss = N_experts * (f * p).sum()
```

**Why multiply f and p?**

`f_i` carries the signal about actual overload; `p_i` carries the gradient pathway.
Multiplying them makes the gradient on each `p_i` proportional to how overloaded expert i is:

```
∂aux_loss / ∂p_i  =  N * f_i
```

- Expert overloaded → high `f_i` → large gradient pushing `p_i` down
- Expert underused  → low  `f_i` → small gradient, leaves `p_i` alone

This is the key design: `f_i` steers where to push while `p_i` is the handle gradients
can actually turn.

**Why multiply by `N_experts`?**

To keep the loss magnitude stable as N changes.
With perfectly balanced routing `f_i = p_i = 1/N`:

```
Σ_i (f_i * p_i)  =  N * (1/N)^2  =  1/N
aux_loss          =  N * (1/N)    =  1.0
```

Without the N multiplier the sum would be `1/N`, shrinking as you add experts.
The N factor normalises so the minimum is always 1.0 regardless of N, making
`aux_loss_coeff` independent of model width.

---

#### What does minimising it do?

| Scenario | f | p | f*p | gradient effect |
|---|---|---|---|---|
| Expert 0 gets 90% of tokens | f_0=0.9 | p_0=0.85 | 0.765 | strong push down on p_0 |
| Expert 1 gets 5% of tokens  | f_1=0.05| p_1=0.07 | 0.004 | weak push on p_1 |
| Perfectly balanced (goal)   | f_i=1/N | p_i=1/N  | 1/N² | uniform, minimum |

The final training loss is:

```
loss = CE_loss + aux_loss_coeff * mean(aux_loss over all MoE blocks)
```

`aux_loss_coeff` (α ≈ 0.01) keeps the balancing signal as a gentle nudge — strong enough to
prevent collapse, weak enough not to fight the main language modelling objective.

---

### MoELayer — How Token Dispatch Actually Works

MoELayer is the drop-in replacement for a dense FFN. Its job is to route each token to its
assigned experts, collect the outputs, and blend them back together.

#### Full forward walkthrough

**Input:** `x` of shape `(B, CL, ED)`

**Step 1 — Flatten**
```python
x_flat = x.view(T, ED)   # T = B * CL
```
The router and experts work on individual tokens, not sequences.
Flattening lets us treat the batch dimension and sequence dimension identically.

**Step 2 — Route**
```python
top_k_weights, top_k_idx, aux_loss = router(x_flat)
# top_k_weights: (T, k)  — renormalised weights, each row sums to 1
# top_k_idx:     (T, k)  — which expert each weight belongs to
```

Example with T=4 tokens, N=4 experts, k=2:
```
top_k_idx     = [[2, 0],    # token 0 → experts 2 and 0
                 [1, 3],    # token 1 → experts 1 and 3
                 [0, 2],    # token 2 → experts 0 and 2
                 [3, 1]]    # token 3 → experts 3 and 1

top_k_weights = [[0.6, 0.4],
                 [0.7, 0.3],
                 [0.5, 0.5],
                 [0.8, 0.2]]
```

**Step 3 — Dispatch loop over experts**

For each expert i we need to:
- find which tokens were sent to it
- run those tokens through the expert
- extract the weight assigned to expert i for each token
- add the weighted output back to the right positions

```python
output = zeros(T, ED)   # accumulation buffer

for i, expert in enumerate(experts):

    # Which tokens listed expert i in their top-k?
    # token_mask_2d[t, ki] = True  if top_k_idx[t, ki] == i
    token_mask_2d = (top_k_idx == i)      # (T, k)  bool
    routed        = token_mask_2d.any(dim=-1)  # (T,)   bool
```

Continuing the example for **expert 0**:
```
top_k_idx == 0:
    token 0: [False, True ]  → routed (expert 0 is at position k=1)
    token 1: [False, False]  → not routed
    token 2: [True,  False]  → routed (expert 0 is at position k=0)
    token 3: [False, False]  → not routed

routed = [True, False, True, False]
```

```python
    expert_out = expert(x_flat[routed])   # (n_i, ED)  — only tokens 0 and 2
```

**Extracting weights:** `top_k_weights[token_mask_2d]` uses the same boolean mask.
Since each routed token has exactly one True in its row, boolean indexing scans row-by-row
and extracts values in token order — no manual reordering needed.

```
top_k_weights[token_mask_2d]:
    token 0 has True at k=1  →  weight 0.4
    token 2 has True at k=0  →  weight 0.5

weights_i = [[0.4],    # (n_i, 1)
             [0.5]]
```

```python
    weights_i = top_k_weights[token_mask_2d].unsqueeze(-1)   # (n_i, 1)
    output[routed] += weights_i * expert_out                  # weighted scatter-add
```

After the loop for all 4 experts, every token has received contributions from its k experts,
each scaled by its router weight.

**Step 4 — Restore shape**
```python
return output.view(B, CL, ED), aux_loss
```

---

### MoETransformerBlock

Identical to a standard pre-norm Transformer block except the FFN is replaced by MoELayer,
and the block returns `(output, aux_loss)` so the aux loss can bubble up to the model.

```
x → LayerNorm → Attention  → residual add  →
  → LayerNorm → MoELayer   → residual add  → (output, aux_loss)
```

---

### GPTModelMoE

Full model wiring:

```
token embeddings (no positional embedding — position handled by RoPE inside attention)
→ dropout
→ N MoETransformerBlocks, accumulating aux_loss at each block
→ final LayerNorm
→ lm_head (weight-tied to token embedding)
```

Loss during training:
```python
mean_aux = total_aux_loss / n_transformer_blocks
loss     = CE_loss + aux_loss_coeff * mean_aux
```

Averaging over blocks (rather than summing) keeps `aux_loss_coeff` meaningful regardless
of model depth — the same α works whether you have 6 or 24 blocks.

During inference (`targetTox=None`) the aux loss is still computed inside the router
but is discarded. Only `logits` for the last position is returned.
