from dataclasses import dataclass, field
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
import math

# 2hrs

@dataclass
class GPTConfig:
    context_len: int = 1024 # aka block_size, sequence_len, etc..
    batch_size: int = 32
    emb_dim: int = 768
    n_heads: int = 12
    n_transformer_blocks: int = 12
    dropout: float = 0.05
    vocab_size: int = 50257 # 50k merges, 256 byte tokes, 1 special eof token
    use_flash_attn: bool = False
    warmup_iters: int = 10
    weight_decay: float = 0.1
    total_batch_size: int = 524288 # 2^19, This is the total no. of tokens we want to train per batch.
    learning_rate: float = 6e-4
    max_train_iters: int = 50
    min_lr: float = 6e-5
    grad_acum_steps: int = field(init=False)

    def __post_init__(self):
        # so we would have to run grad_acum_steps no. of steps every time before we update the model params (i.e optim.step()).
        # So we would run `batch_size'd` itr every time and accumilate the grads till `grad_acum_steps` and then finally do optim.step
        # So batch_size in a way becomes a micro batch in every step
        self.grad_acum_steps = self.total_batch_size // (self.batch_size * self.context_len) # would come to 64 if batch_size is 8


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        #k,q,v matrix combined
        self.c_attn = nn.Linear(config.emb_dim, config.emb_dim * 3)
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim)

        self.n_embd = config.emb_dim
        self.n_head = config.n_heads
        self.use_flash_attn = config.use_flash_attn

        #For some reason gpt2 impl called it bias, so keeping this as bias, its just a mask
        # We need to change the shape so that it works with (B, n_head, CL, ED)
        self.register_buffer('bias', torch.tril(torch.ones(config.context_len, config.context_len)).view(1,1,config.context_len,config.context_len))

    def forward(self, x: torch.Tensor): # (B, CL, ED)
        B, CL, ED = x.size()
        qkvx = self.c_attn(x) #(B, CL, 3*ED)
        qx,kx,vx = qkvx.split(self.n_embd, dim=2)

        # Now split into heads
        head_dim = self.n_embd // self.n_head
        qx = qx.view(B, CL, self.n_head, head_dim) #(B, CL, n_head, head_dim)
        kx = kx.view(B, CL, self.n_head, head_dim) #(B, CL, n_head, head_dim)
        vx = vx.view(B, CL, self.n_head, head_dim) #(B, CL, n_head, head_dim)

        # Transpose to make it work efficiently with pytorch, essentially making it (B, n_head, CL, head_dim) which is each head has its set of features and can work parallelly
        qx = qx.transpose(1, 2) #(B, n_head, CL, head_dim)
        kx = kx.transpose(1, 2) #(B, n_head, CL, head_dim)
        vx = vx.transpose(1, 2) #(B, n_head, CL, head_dim)

        if self.use_flash_attn:
            attn_w = F.scaled_dot_product_attention(qx, kx, vx, is_causal=True) # (B, n_head, CL, head_dim)
        else:
            attn = qx @ kx.transpose(-2, -1) * (head_dim ** -0.5) # (B, n_head, CL, CL)
            attn = attn.masked_fill(self.bias[:,:,:CL,:CL] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn_w = attn @ vx # (B, n_head, CL, head_dim)
        attn_w = attn_w.transpose(1, 2) # (B, CL, n_head, head_dim)
        attn_w = attn_w.reshape(B, CL, ED) # (B, CL, ED)

        out = self.c_proj(attn_w)
        return out


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.emb_dim, config.emb_dim * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.emb_dim * 4, config.emb_dim)

    def forward(self, x): # (B, CL, ED)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x # (B, CL, ED)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # LayerNorm1 => Attn => LayerNorm2 => MLP
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)

    def forward(self, x): #(B, CL, ED)
        # NOTE: This differs from the original attn paper that layer norm is applied before attn and MLP
        # Also the residual stream is clean before it gets added to the attn/MLP output unlike original attn paper

        # Think of attn is where communicate with each other (tokens from past)
        # This is like an pooling fuction
        x = x + self.attn(self.ln_1(x))

        # MLP is where each token operates individually on the info they got from attn layer
        x = x + self.mlp(self.ln_2(x))
        return x


# Here we are trying to exactly replicate the Huggingface GPT 2 model architecture
# Most of the variable names would be similar
class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.emb_dim),
                wpe = nn.Embedding(config.context_len, config.emb_dim),

                # These are hidden layer or transformer blocks
                h = nn.ModuleList(Block(config) for _ in range(config.n_transformer_blocks)),

                # Final layer norm after attn and MLP
                ln_f = nn.LayerNorm(config.emb_dim)
            )
        )

        # Unembedding — no bias, weight tied to wte (same matrix, counted once)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        # Residual projections (c_proj) get a smaller std to keep the residual stream stable at init.
        # std = 0.02 / sqrt(2 * n_layers). See NOTES.md § Variance Accumulation → Scaled Residual Init
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_transformer_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None): #(B, CL)
        _, CL = x.shape
        tok_emb = self.transformer.wte(x) #(B, CL, ED)
        pos_emb = self.transformer.wpe(torch.arange(0, CL, dtype=torch.long, device=x.device))
        y = tok_emb + pos_emb
        for block in self.transformer.h:
            y = block(y) # (B, CL, ED)

        y = self.transformer.ln_f(y) # (B, CL, ED)
        logits = self.lm_head(y) # (B, CL, V)
        loss = None

        if targets is not None:
            # We need to flatten the logits and targets as cross_entropy only takes 2D arrays
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    ## Support for loading weights from HF gpt2 model
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"Loading pretrained gpt-2 model: {model_type}")
        config_args = {
            "gpt2":        dict(vocab_size=50257, context_len=1024, emb_dim=768,  n_heads=12, n_transformer_blocks=12),
            "gpt2-medium": dict(vocab_size=50257, context_len=1024, emb_dim=1024, n_heads=16, n_transformer_blocks=24),
            "gpt2-large":  dict(vocab_size=50257, context_len=1024, emb_dim=1280, n_heads=20, n_transformer_blocks=36),
            "gpt2-xl":     dict(vocab_size=50257, context_len=1024, emb_dim=1600, n_heads=25, n_transformer_blocks=48),
        }[model_type]
        config = GPTConfig(**config_args)
        model = GPT2(config)
        m_sd = model.state_dict()
        m_sd_keys = list(filter(lambda k: not k.endswith('.attn.bias'), m_sd.keys())) # We don't need attn mask (the tril which we called as bias)

        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_model_sd = hf_model.state_dict()
        hf_model_sd_keys = list(filter(
            lambda k: not k.endswith('.attn.bias') and not k.endswith('.attn.masked_bias'),
            hf_model_sd.keys()
        ))

        # openai checkpoints use a "Conv1D" module (not nn.Conv1d, HF internal module where weights shape is (in, out); nn.Linear is declared as (out, in))
        # Hence we transpose these weights before copying
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        print(f"Len of model keys {len(m_sd_keys)}, len of hf model keys {len(hf_model_sd_keys)}")
        assert len(hf_model_sd_keys) == len(m_sd_keys), f"mismatched keys: {len(hf_model_sd_keys)} != {len(m_sd_keys)}"
        for k in hf_model_sd_keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # ::-1 reverses the shape
                assert hf_model_sd[k].shape[::-1] == m_sd[k].shape
                with torch.no_grad():
                    m_sd[k].copy_(hf_model_sd[k].t())
            else:
                # vanilla copy over the other parameters
                assert hf_model_sd[k].shape == m_sd[k].shape
                with torch.no_grad():
                    m_sd[k].copy_(hf_model_sd[k])
        return model

    @classmethod
    def sample(cls, model, start_text, max_len, n_samples, seed=None,
               temperature=1.0, top_k=None):
        """Generate text from the model.

        Args:
            model:       GPT2 model (stays on whatever device it's already on)
            start_text:  prompt string
            max_len:     max tokens to generate (including the prompt)
            n_samples:   how many independent completions to produce
            seed:        optional RNG seed for reproducibility
            temperature: <1 = sharper/more deterministic, >1 = more random, 1.0 = raw softmax
            top_k:       if set, only sample from the top-k most probable tokens
        """
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        enc = tiktoken.get_encoding("gpt2")
        inp_toks = torch.tensor(enc.encode(start_text), dtype=torch.long)
        inp_toks = inp_toks.unsqueeze(0).repeat(n_samples, 1)  # (n_samples, seq_len)
        x_gen = inp_toks.to(device)

        rng = None
        if seed is not None:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        while x_gen.size(1) <= max_len:
            with torch.no_grad():
                logits, _ = model(x_gen[:, -1024:])  # (n_samples, seq_len, vocab_size)
                logits = logits[:, -1, :]             # (n_samples, vocab_size)

                # Temperature: divide logits before softmax to control randomness
                if temperature != 1.0:
                    logits = logits / temperature

                # Top-k: zero out everything outside the top-k tokens so they can't be sampled
                if top_k is not None:
                    top_vals, _ = torch.topk(logits, top_k, dim=-1)
                    logits[logits < top_vals[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=rng)
                x_gen = torch.cat([x_gen, next_token], dim=1)

        if was_training:
            model.train()

        results = []
        for i in range(n_samples):
            text = enc.decode(x_gen[i, :].tolist())
            results.append(text)
            print(text)
        return results


class DataLoader:
    def __init__(self, raw_txt, batch_size, context_len, num_processes=1, process_rank=0,
                 tokenizer=None, split="train", train_ratio=0.9, val_ratio=0.05):
        tkzer = tokenizer
        if tkzer is None:
            tkzer = tiktoken.get_encoding("gpt2")

        all_tokens = torch.tensor(tkzer.encode(raw_txt))
        n = len(all_tokens)

        # Split the token stream into train / val / test by ratio.
        # Tokens are contiguous — no shuffling, so each split sees a different part of the text.
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

        print(f"DataLoader [{split}]: {len(self.data):,} tokens (of {n:,} total)")
        self.batch_size = batch_size
        self.context_len = context_len
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Each process starts at a different offset so they read non-overlapping slices of the dataset.
        self.current_pos = self.batch_size * self.context_len * self.process_rank
        print(f"  batches per epoch = {len(self.data) // (self.batch_size * self.context_len)}")

    def total_tokens(self) -> int:
        return len(self.data)

    def get_batch(self):
        # +1 so that buf[:-1] and buf[1:] both have exactly batch_size * context_len tokens
        buf = self.data[self.current_pos : self.current_pos + self.batch_size * self.context_len + 1]
        inputs  = (buf[:-1]).view(self.batch_size, self.context_len) #(B, CL) => current to N
        targets = (buf[1:]).view(self.batch_size, self.context_len)  #(B, CL) => current+1 to N+1

        # Advance by the full global batch width so this process's next chunk skips over the slices owned by all other processes.
        self.current_pos += self.batch_size * self.context_len * self.num_processes
        # Reset the position if we are at end
        if self.current_pos + (self.batch_size * self.context_len + 1) > len(self.data):
            self.current_pos = self.batch_size * self.context_len * self.process_rank
        return inputs, targets


class ShardedDataLoader:
    """Reads pre-tokenized .npy shards from multiple datasets and mixes them by weight.

    Each dataset is a directory of shard_NNNNNN.npy files produced by prepare_data.py.
    On every get_batch() call, a dataset is chosen randomly according to the mixing
    weights, and a contiguous batch is read from that dataset's current shard position.

    Why shards?
      The full dataset (e.g. FineWeb-Edu 10BT ≈ 20 GB of uint16) doesn't fit in RAM.
      We split it into fixed-size shards (~100M tokens / ~200 MB each) so only one shard
      per dataset needs to be resident at a time.

    Mixing strategy (batch-level):
      Each get_batch() call picks ONE dataset using random.choices (weighted).
      Over many steps the proportion of batches from each dataset converges to the
      target weights (e.g. 55 / 25 / 20). This is simpler than token-level interleaving
      and works well in practice.
    """

    def __init__(self, data_dir: str, dataset_weights: dict[str, float],
                 batch_size: int, context_len: int,
                 split: str = "train",
                 num_processes: int = 1, process_rank: int = 0):
        self.batch_size = batch_size
        self.context_len = context_len
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.tokens_per_batch = batch_size * context_len
        self.split = split

        # dataset_weights: {"fineweb": 0.55, "books": 0.25, "tinystories": 0.20}
        self.ds_names = list(dataset_weights.keys())
        self.weights = [dataset_weights[n] for n in self.ds_names]

        # Per-dataset state: which shard is loaded, where we are within it.
        # Only one shard per dataset is in memory at a time.
        # Shard filenames are prefixed by split: train_shard_*.npy, val_shard_*.npy, test_shard_*.npy
        # (created by prepare_data.assign_splits)
        self.ds_state: dict[str, dict] = {}
        self.ds_token_counts: dict[str, int] = {}
        for name in self.ds_names:
            shard_dir = os.path.join(data_dir, name)
            shards = sorted(glob.glob(os.path.join(shard_dir, f"{split}_shard_*.npy")))
            assert len(shards) > 0, f"No {split} shards found for dataset '{name}' in {shard_dir}"

            # Shards are uint16 on disk; cast to int64 for torch embedding lookups
            self.ds_state[name] = {
                "shards": shards,
                "shard_idx": 0,
                # In DDP each rank starts at a different offset so they read non-overlapping slices
                "pos": self.tokens_per_batch * process_rank,
                "data": torch.from_numpy(np.load(shards[0]).astype(np.int64)),
            }
            total = sum(len(np.load(s)) for s in shards)
            self.ds_token_counts[name] = total
            print(f"ShardedDataLoader [{split}]: '{name}' — {len(shards)} shard(s), {total:,} tokens")

    def _advance_shard(self, state: dict):
        """Move to the next shard (wraps around to the first shard after the last)."""
        state["shard_idx"] = (state["shard_idx"] + 1) % len(state["shards"])
        state["data"] = torch.from_numpy(
            np.load(state["shards"][state["shard_idx"]]).astype(np.int64)
        )
        # Reset position — each DDP rank gets its own starting offset within the new shard
        state["pos"] = self.tokens_per_batch * self.process_rank

    def total_tokens(self) -> dict[str, int]:
        """Return per-dataset and combined token counts.

        Example output:
          {"fineweb": 5_000_000_000, "books": 800_000_000, "tinystories": 400_000_000, "total": 6_200_000_000}
        """
        result = dict(self.ds_token_counts)
        result["total"] = sum(self.ds_token_counts.values())
        return result

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick a dataset according to the mixing weights (batch-level mixing)
        ds_name = random.choices(self.ds_names, weights=self.weights, k=1)[0]
        state = self.ds_state[ds_name]

        # +1 because targets are shifted by one token: inputs = buf[:-1], targets = buf[1:]
        needed = self.tokens_per_batch + 1
        if state["pos"] + needed > len(state["data"]):
            self._advance_shard(state)

        buf = state["data"][state["pos"] : state["pos"] + needed]
        inputs  = buf[:-1].view(self.batch_size, self.context_len)  # (B, CL)
        targets = buf[1:].view(self.batch_size, self.context_len)   # (B, CL)

        # Skip over slices owned by other DDP processes
        state["pos"] += self.tokens_per_batch * self.num_processes
        return inputs, targets