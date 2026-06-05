"""
GPT-2 model, data loaders, and config for Apple MLX.

MLX equivalent of gpt2_model.py (PyTorch). Key differences from PyTorch:
  - Single Apple Silicon device — no DDP / multi-GPU support
  - mx.array instead of torch.Tensor; int32 for embedding indices (not int64)
  - nn.Module uses __call__ instead of forward()
  - Causal mask computed on-the-fly (no register_buffer needed)
  - Weight tying: explicit matmul y @ wte.weight.T (no separate lm_head parameter)
  - cross_entropy needs explicit float32 cast (MLX has no autocast)
  - Lazy evaluation: computations aren't executed until mx.eval() is called
"""

from dataclasses import dataclass, field
import os
import glob
import random
import mlx.core as mx
import mlx.nn as nn
import tiktoken
import numpy as np
import math


@dataclass
class GPTConfig:
    context_len: int = 1024    # aka block_size, sequence_len
    batch_size: int = 32
    emb_dim: int = 768
    n_heads: int = 12
    n_transformer_blocks: int = 12
    dropout: float = 0.05
    vocab_size: int = 50257    # 50k merges + 256 byte tokens + 1 special eof token
    use_flash_attn: bool = False
    warmup_iters: int = 10
    weight_decay: float = 0.1
    total_batch_size: int = 524288  # 2^19, total tokens per training step
    learning_rate: float = 6e-4
    max_train_iters: int = 50
    min_lr: float = 6e-5
    grad_acum_steps: int = field(init=False)

    def __post_init__(self):
        # How many micro-batches to accumulate before an optimizer step.
        # Each micro-batch processes batch_size * context_len tokens, so we need
        # grad_acum_steps micro-batches to reach total_batch_size tokens per step.
        self.grad_acum_steps = self.total_batch_size // (self.batch_size * self.context_len)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.emb_dim, config.emb_dim * 3)
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.n_embd = config.emb_dim
        self.n_head = config.n_heads
        self.use_flash_attn = config.use_flash_attn

    def __call__(self, x):  # (B, CL, ED)
        B, CL, ED = x.shape
        qkvx = self.c_attn(x)  # (B, CL, 3*ED)
        qx, kx, vx = mx.split(qkvx, 3, axis=2)  # each (B, CL, ED)

        head_dim = self.n_embd // self.n_head
        qx = qx.reshape(B, CL, self.n_head, head_dim).transpose(0, 2, 1, 3)  # (B, n_head, CL, head_dim)
        kx = kx.reshape(B, CL, self.n_head, head_dim).transpose(0, 2, 1, 3)
        vx = vx.reshape(B, CL, self.n_head, head_dim).transpose(0, 2, 1, 3)

        if self.use_flash_attn:
            # mx.fast.scaled_dot_product_attention handles causal masking internally
            attn_w = mx.fast.scaled_dot_product_attention(qx, kx, vx, scale=head_dim**-0.5, mask='causal')
        else:
            attn = (qx @ kx.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)  # (B, n_head, CL, CL)
            # Compute mask on-the-fly — avoids storing it as a tracked parameter
            # (PyTorch uses register_buffer for this instead)
            causal_mask = mx.tril(mx.ones((CL, CL), dtype=mx.bool_))
            attn = mx.where(causal_mask, attn, mx.array(float('-inf')))
            attn = mx.softmax(attn, axis=-1)
            attn_w = attn @ vx  # (B, n_head, CL, head_dim)

        attn_w = attn_w.transpose(0, 2, 1, 3).reshape(B, CL, ED)  # (B, CL, ED)
        return self.c_proj(attn_w)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.emb_dim, config.emb_dim * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.emb_dim * 4, config.emb_dim)

    def __call__(self, x):  # (B, CL, ED)
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)

    def __call__(self, x):  # (B, CL, ED)
        # Pre-norm architecture (LayerNorm before attn/MLP, not after)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.wpe = nn.Embedding(config.context_len, config.emb_dim)
        self.h = [Block(config) for _ in range(config.n_transformer_blocks)]
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self._init_weights(config)

    def _init_weights(self, config):
        # GPT-2 paper: normal(0, 0.02); c_proj layers scaled down by sqrt(2 * n_layers)
        # to keep the residual stream stable at init (see NOTES.md)
        std = 0.02
        c_proj_std = std / math.sqrt(2 * config.n_transformer_blocks)

        self.wte.weight = mx.random.normal(shape=self.wte.weight.shape) * std
        self.wpe.weight = mx.random.normal(shape=self.wpe.weight.shape) * std

        for block in self.h:
            block.attn.c_attn.weight = mx.random.normal(shape=block.attn.c_attn.weight.shape) * std
            block.attn.c_attn.bias = mx.zeros(block.attn.c_attn.bias.shape)
            block.attn.c_proj.weight = mx.random.normal(shape=block.attn.c_proj.weight.shape) * c_proj_std
            block.attn.c_proj.bias = mx.zeros(block.attn.c_proj.bias.shape)
            block.mlp.c_fc.weight = mx.random.normal(shape=block.mlp.c_fc.weight.shape) * std
            block.mlp.c_fc.bias = mx.zeros(block.mlp.c_fc.bias.shape)
            block.mlp.c_proj.weight = mx.random.normal(shape=block.mlp.c_proj.weight.shape) * c_proj_std
            block.mlp.c_proj.bias = mx.zeros(block.mlp.c_proj.bias.shape)

    def __call__(self, x, targets=None):  # x: (B, CL)
        B, CL = x.shape
        tok_emb = self.wte(x)                        # (B, CL, ED)
        pos_emb = self.wpe(mx.arange(CL))            # (CL, ED)
        y = tok_emb + pos_emb
        for block in self.h:
            y = block(y)
        y = self.ln_f(y)
        # Weight-tied unembedding — reuses wte.weight, no separate lm_head parameter.
        # PyTorch version uses an explicit nn.Linear(lm_head) with weight = wte.weight.
        logits = y @ self.wte.weight.T               # (B, CL, vocab_size)

        loss = None
        if targets is not None:
            # Cast logits to float32 before cross_entropy.
            #
            # Why needed here but NOT in PyTorch:
            #   PyTorch's torch.autocast maintains an internal whitelist of ops that must
            #   stay in float32 — cross_entropy is on that list. So even inside autocast(bfloat16),
            #   PyTorch silently upcasts logits before computing the loss.
            #
            #   MLX has no autocast — we cast weights globally via set_dtype(), so there's no
            #   runtime op-level dtype policy. Logits come out as bfloat16 and MLX's cross_entropy
            #   runs in whatever dtype it receives.
            #
            # Why bfloat16 is bad for cross_entropy:
            #   cross_entropy sums exp(logit_i) over all 50k vocab tokens. bfloat16 only has 7
            #   mantissa bits (~2 decimal digits), so the softmax denominator is very coarse.
            #   This causes loss to snap to bfloat16-representable values (11.0, 9.5, 8.75...)
            #   instead of true values, giving the optimizer coarser gradient signal.
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
                targets.reshape(-1),
                reduction='mean'
            )
        return logits, loss

    @classmethod
    def sample(cls, model, start_text, max_len, n_samples, seed=None,
               temperature=1.0, top_k=None):
        """Generate text from the model.

        Args:
            model:       GPT2 model instance
            start_text:  prompt string
            max_len:     max tokens to generate (including the prompt)
            n_samples:   how many independent completions to produce
            seed:        optional RNG seed for reproducibility
            temperature: <1 = sharper/more deterministic, >1 = more random, 1.0 = raw softmax
            top_k:       if set, only sample from the top-k most probable tokens
        """
        if seed is not None:
            mx.random.seed(seed)

        enc = tiktoken.get_encoding("gpt2")
        inp_toks = mx.array(enc.encode(start_text))
        x_gen = mx.stack([inp_toks] * n_samples)      # (n_samples, seq_len)

        while x_gen.shape[1] <= max_len:
            logits, _ = model(x_gen[:, -1024:])        # (n_samples, seq_len, vocab_size)
            logits = logits[:, -1, :]                  # (n_samples, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature

            # Top-k: zero out everything outside the top-k tokens so they can't be sampled
            if top_k is not None:
                top_vals = mx.topk(logits, k=top_k, axis=-1)      # (n_samples, top_k)
                threshold = mx.min(top_vals, axis=-1, keepdims=True)  # smallest of top-k
                logits = mx.where(logits < threshold, float('-inf'), logits)

            probs = mx.softmax(logits, axis=-1)
            # mx.random.categorical expects log-probabilities
            next_token = mx.random.categorical(mx.log(probs))      # (n_samples,)
            x_gen = mx.concatenate([x_gen, next_token[:, None]], axis=1)
            mx.eval(x_gen)

        results = []
        for i in range(n_samples):
            text = enc.decode(x_gen[i].tolist())
            results.append(text)
            print(text)
        return results


class DataLoader:
    """Tokenizes raw text and serves batches from a specific split.

    Splits the token stream into train/val/test by ratio (contiguous, no shuffling).
    MLX equivalent of DataLoader in gpt2_model.py — no DDP support since
    MLX runs on a single Apple Silicon device.
    """

    def __init__(self, raw_txt, batch_size, context_len, tokenizer=None,
                 split="train", train_ratio=0.9, val_ratio=0.05):
        tkzer = tokenizer or tiktoken.get_encoding("gpt2")
        all_tokens = mx.array(tkzer.encode(raw_txt))
        n = all_tokens.shape[0]

        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.data = all_tokens[:train_end]
        elif split == "val":
            self.data = all_tokens[train_end:val_end]
        elif split == "test":
            self.data = all_tokens[val_end:]
        else:
            raise ValueError(f"Unknown split '{split}', expected 'train', 'val', or 'test'")

        print(f"DataLoader [{split}]: {self.data.shape[0]:,} tokens (of {n:,} total)")
        self.batch_size = batch_size
        self.context_len = context_len
        self.current_pos = 0
        print(f"  batches per epoch = {self.data.shape[0] // (self.batch_size * self.context_len)}")

    def total_tokens(self) -> int:
        return self.data.shape[0]

    def get_batch(self):
        buf = self.data[self.current_pos : self.current_pos + self.batch_size * self.context_len + 1]
        inputs  = buf[:-1].reshape(self.batch_size, self.context_len)   # (B, CL)
        targets = buf[1:].reshape(self.batch_size, self.context_len)    # (B, CL)
        self.current_pos += self.batch_size * self.context_len
        if self.current_pos + (self.batch_size * self.context_len + 1) > self.data.shape[0]:
            self.current_pos = 0
        return inputs, targets


class ShardedDataLoader:
    """Reads pre-tokenized .npy shards from multiple datasets and mixes them by weight.

    MLX equivalent of ShardedDataLoader in gpt2_model.py (PyTorch).
    No DDP support needed — MLX runs on a single Apple Silicon device.

    Each dataset is a directory of {split}_shard_NNNNNN.npy files produced by prepare_data.py.
    On every get_batch() call, a dataset is chosen randomly according to mixing weights,
    and a contiguous batch is read from that dataset's current shard position.

    Shards are uint16 on disk; cast to int32 for MLX embedding lookups
    (MLX embeddings expect int32, PyTorch uses int64).
    """

    def __init__(self, data_dir: str, dataset_weights: dict[str, float],
                 batch_size: int, context_len: int, split: str = "train"):
        self.batch_size = batch_size
        self.context_len = context_len
        self.tokens_per_batch = batch_size * context_len
        self.split = split

        self.ds_names = list(dataset_weights.keys())
        self.weights = [dataset_weights[n] for n in self.ds_names]

        self.ds_state: dict[str, dict] = {}
        self.ds_token_counts: dict[str, int] = {}
        for name in self.ds_names:
            shard_dir = os.path.join(data_dir, name)
            shards = sorted(glob.glob(os.path.join(shard_dir, f"{split}_shard_*.npy")))
            assert len(shards) > 0, f"No {split} shards found for dataset '{name}' in {shard_dir}"

            self.ds_state[name] = {
                "shards": shards,
                "shard_idx": 0,
                "pos": 0,
                "data": mx.array(np.load(shards[0]).astype(np.int32)),
            }
            total = sum(len(np.load(s)) for s in shards)
            self.ds_token_counts[name] = total
            print(f"ShardedDataLoader [{split}]: '{name}' — {len(shards)} shard(s), {total:,} tokens")

    def _advance_shard(self, state: dict):
        state["shard_idx"] = (state["shard_idx"] + 1) % len(state["shards"])
        state["data"] = mx.array(
            np.load(state["shards"][state["shard_idx"]]).astype(np.int32)
        )
        state["pos"] = 0

    def total_tokens(self) -> dict[str, int]:
        result = dict(self.ds_token_counts)
        result["total"] = sum(self.ds_token_counts.values())
        return result

    def get_batch(self) -> tuple:
        # Pick a dataset according to mixing weights (batch-level mixing)
        ds_name = random.choices(self.ds_names, weights=self.weights, k=1)[0]
        state = self.ds_state[ds_name]

        # +1 because targets are shifted by one token: inputs = buf[:-1], targets = buf[1:]
        needed = self.tokens_per_batch + 1
        if state["pos"] + needed > state["data"].shape[0]:
            self._advance_shard(state)

        buf = state["data"][state["pos"] : state["pos"] + needed]
        inputs  = buf[:-1].reshape(self.batch_size, self.context_len)   # (B, CL)
        targets = buf[1:].reshape(self.batch_size, self.context_len)    # (B, CL)
        state["pos"] += self.tokens_per_batch
        return inputs, targets
