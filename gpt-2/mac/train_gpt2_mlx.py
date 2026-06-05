"""
GPT-2 training script for Apple MLX.

MLX equivalent of train_gpt2.py (PyTorch). Key differences:
  - No DDP — MLX runs on a single Apple Silicon device
  - nn.value_and_grad for functional gradient computation (vs loss.backward())
  - mx.eval for lazy evaluation — MLX builds a compute graph, eval forces execution
  - Weight decay applies to ALL params (MLX AdamW has no param groups like PyTorch)
  - Checkpoint is a directory (model.safetensors + meta.json), not a single .pt file
  - Optimizer state not saved in checkpoints — standard MLX practice, momentum restarts on resume

Usage:
  python train_gpt2_mlx.py                          # default mode (tinyshakespeare)
  python train_gpt2_mlx.py --sharded                # sharded mode (FineWeb + books + TinyStories)
  python train_gpt2_mlx.py --sharded --max_iters 50
  python train_gpt2_mlx.py --resume                 # resume from checkpoint

Deps:
  pip install mlx tiktoken numpy
"""

import os
import json
import math
import time
import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from gpt2_mlx import GPT2, GPTConfig, DataLoader, ShardedDataLoader


class GPTTrainer:
    def __init__(self, model: GPT2, config: GPTConfig, dataLoader,
                 val_loader=None, test_loader=None, eval_interval: int = 100, eval_batches: int = 20,
                 checkpoint_interval: int = 100, checkpoint_path: str = "out/ckpt_train"):
        self.model = model
        self.config = config
        self.dataLoader = dataLoader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.start_step = 0

        # MLX AdamW — weight_decay applies to all params uniformly.
        # PyTorch version uses param groups to skip decay on biases/LayerNorm;
        # the effect of decaying those small params is minimal in practice.
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=config.weight_decay,
        )

        total_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        print(f"Total trainable parameters: {total_params:,}")

        # nn.value_and_grad: returns a function that computes both the loss AND
        # gradients w.r.t. model parameters in one pass.
        # Created ONCE here so mx.compile can reuse the cached graph across steps.
        self.loss_and_grad_fn = nn.value_and_grad(
            model, lambda m, x, y: m(x, y)[1]
        )

        # ── mx.compile setup ─────────────────────────────────────────────────
        # mx.compile traces a Python function once, builds a fused Metal compute graph,
        # and replays it on every subsequent call — skipping Python overhead entirely.
        # This is the biggest MLX perf lever (~2-4× faster than uncompiled).
        #
        # We can only compile when grad_acum_steps == 1 because mx.compile can't trace
        # through the Python for-loop that accumulates gradients across micro-batches.
        # When grad_acum_steps > 1, we fall back to an uncompiled accumulation loop.
        #
        # inputs/outputs: mx.compile tracks mutable state via array dicts, not Module
        # objects. We pass model.state and optimizer.state so the compiler sees which
        # arrays are read at the start and written back at the end of each call.
        self.compiled_step = None
        if config.grad_acum_steps == 1:
            def step(x, y):
                loss, grads = self.loss_and_grad_fn(model, x, y)
                grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
                self.optimizer.update(model, grads)
                return loss

            self._compile_state = [model.state, self.optimizer.state]
            self.compiled_step = mx.compile(
                step, inputs=self._compile_state, outputs=self._compile_state
            )
            print("Using mx.compile (grad_acum_steps == 1)")
        else:
            print(f"Skipping mx.compile (grad_acum_steps = {config.grad_acum_steps} > 1, using accumulation loop)")

    def get_lr(self, itr):
        # Linear warmup for warmup_iters, then cosine decay to min_lr.
        # Same schedule as PyTorch version (see train_gpt2.py).
        if itr < self.config.warmup_iters:
            return self.config.learning_rate * (itr + 1) / self.config.warmup_iters

        if itr > self.config.max_train_iters:
            return self.config.min_lr

        decay_ratio = (itr - self.config.warmup_iters) / (self.config.max_train_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def evaluate(self, eval_loader, n_batches: int = None) -> float:
        """Run n_batches from eval_loader and return the average loss."""
        if n_batches is None:
            n_batches = self.eval_batches
        total_loss = 0.0
        for _ in range(n_batches):
            x, y = eval_loader.get_batch()
            _, loss = self.model(x, y)
            mx.eval(loss)
            total_loss += loss.item()
        return total_loss / n_batches

    def save_checkpoint(self, step, val_loss=None, inference_only=False):
        """Save model weights and metadata to a directory.

        Unlike the PyTorch version (single .pt file), MLX checkpoints are directories:
          ckpt_train/
            model.safetensors   — model weights
            meta.json           — step, val_loss, config

        Optimizer state is NOT saved — standard MLX practice. On resume, the optimizer's
        running averages (Adam momentum/variance) restart from zero. This causes a brief
        warmup period but converges quickly.
        """
        if inference_only:
            ckpt_dir = self.checkpoint_path + "_inference"
        else:
            ckpt_dir = self.checkpoint_path
        os.makedirs(ckpt_dir, exist_ok=True)

        weights_path = os.path.join(ckpt_dir, "model.safetensors")
        self.model.save_weights(weights_path)

        meta = {
            "step": step,
            "val_loss": val_loss,
            "config": {k: v for k, v in vars(self.config).items()},
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        total_size = sum(
            os.path.getsize(os.path.join(ckpt_dir, f))
            for f in os.listdir(ckpt_dir)
            if os.path.isfile(os.path.join(ckpt_dir, f))
        )
        print(f"Checkpoint saved: {ckpt_dir}  ({total_size / (1024 * 1024):.1f} MB)")

    def load_checkpoint(self, path=None):
        """Load model weights and metadata from a checkpoint directory.

        Args:
            path: Checkpoint directory. Defaults to self.checkpoint_path.

        Optimizer state is not restored — it starts fresh (standard MLX practice).
        """
        path = path or self.checkpoint_path

        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)

        self.model.load_weights(os.path.join(path, "model.safetensors"))
        print("Optimizer state not saved in MLX checkpoints — optimizer starts fresh")

        self.start_step = meta["step"] + 1
        print(f"Resumed from {path} — starting at step {self.start_step}")

    def train_model(self, time_itr=False):
        """Main training loop with LR scheduling, eval, and checkpointing.

        Two execution paths:
          1. compiled (grad_acum_steps == 1): entire forward + backward + clip + update
             is fused into one Metal compute graph via mx.compile. ~2-4× faster.
          2. uncompiled (grad_acum_steps > 1): Python loop accumulates gradients over
             micro-batches. mx.eval called per micro-step to bound memory.
        """
        itrs = self.config.max_train_iters
        use_compiled = self.compiled_step is not None

        for step in range(self.start_step, itrs):
            t0 = time.time()

            # ── Validation check (every eval_interval steps + step 0 for baseline) ──
            if self.val_loader and step % self.eval_interval == 0:
                val_loss = self.evaluate(self.val_loader)
                print(f"Step {step}: val_loss = {val_loss:.4f}")

            # Cosine LR schedule with linear warmup
            lr = self.get_lr(step)
            self.optimizer.learning_rate = lr

            if use_compiled:
                # ── Fast path: mx.compile fuses the whole step into one Metal graph ──
                x, y = self.dataLoader.get_batch()
                loss = self.compiled_step(x, y)
                # mx.eval through the declared state outputs — this is what the compiler tracks.
                # Including loss prevents the graph from accumulating across steps.
                mx.eval(self._compile_state, loss)
                acc_loss = loss.item()
                norm = None
            else:
                # ── Accumulation path: Python loop over micro-batches ──
                acc_loss = 0.0
                acc_grads = None

                for _ in range(self.config.grad_acum_steps):
                    x, y = self.dataLoader.get_batch()
                    loss, grads = self.loss_and_grad_fn(self.model, x, y)

                    if acc_grads is None:
                        acc_grads = grads
                    else:
                        acc_grads = tree_map(lambda a, b: a + b, acc_grads, grads)

                    # Force evaluation each micro-step to prevent the lazy compute graph
                    # from growing unboundedly (would OOM for large grad_acum_steps).
                    mx.eval(loss, acc_grads)
                    acc_loss += loss.item()

                # Average gradients across micro-steps
                acc_grads = tree_map(lambda g: g * (1.0 / self.config.grad_acum_steps), acc_grads)
                acc_loss /= self.config.grad_acum_steps

                acc_grads, norm = optim.clip_grad_norm(acc_grads, max_norm=1.0)
                self.optimizer.update(self.model, acc_grads)
                mx.eval(self.model.state, self.optimizer.state, norm)

            if step < 10 and time_itr:
                t1 = time.time()
                dt = (t1 - t0) * 1000
                tokens_processed = self.config.batch_size * self.config.context_len * self.config.grad_acum_steps
                token_throughput = tokens_processed / (t1 - t0)
                norm_str = f"{norm.item():.4f}" if norm is not None else "n/a"
                print(f"Step {step},  loss = {acc_loss:.4f} | dt = {dt:.2f} ms | norm = {norm_str} | LR = {lr:.2e} | token_throughput = {token_throughput:.2f} tokens/s")

            # ── Checkpoint (every checkpoint_interval steps) ──
            if self.checkpoint_interval and (step + 1) % self.checkpoint_interval == 0:
                ckpt_val_loss = self.evaluate(self.val_loader) if self.val_loader else None
                self.save_checkpoint(step, val_loss=ckpt_val_loss)

        # ── Final test evaluation ─────────────────────────────────────────────
        if self.test_loader:
            test_loss = self.evaluate(self.test_loader)
            print(f"\nTest loss = {test_loss:.4f}")

        # Save final checkpoints — full + inference-only
        final_val_loss = self.evaluate(self.val_loader) if self.val_loader else None
        self.save_checkpoint(itrs - 1, val_loss=final_val_loss)
        self.save_checkpoint(itrs - 1, val_loss=final_val_loss, inference_only=True)


if __name__ == "__main__":
    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train GPT-2 from scratch (MLX)")
    parser.add_argument("--sharded", action="store_true",
                        help="Use ShardedDataLoader (pre-tokenized .npy shards from prepare_data.py) "
                             "instead of the default single-file DataLoader")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to shard directory (default: <script_dir>/data). Only used with --sharded")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Number of training iterations (default: 10)")
    parser.add_argument("--checkpoint_path", type=str, default="out/ckpt_train",
                        help="Directory path for saving/loading checkpoints (default: out/ckpt_train)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from --checkpoint_path")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save a checkpoint every N steps (default: 100)")
    parser.add_argument("--bfloat16", action="store_true",
                        help="Cast model weights to bfloat16 before training (saves memory, faster on Apple Silicon)")
    args = parser.parse_args()

    MAX_ITRS = args.max_iters

    config = GPTConfig(
        context_len=1024,
        batch_size=8,
        emb_dim=768,
        n_heads=12,
        n_transformer_blocks=12,
        dropout=0.05,
        vocab_size=50304,       # padded to nearest multiple of 128 for efficiency
        use_flash_attn=False,
        warmup_iters=10,
        learning_rate=6e-4,
        max_train_iters=MAX_ITRS,
        min_lr=6e-5,            # 10% of max LR
        weight_decay=0.1,
        total_batch_size=524288,  # 2^19
    )
    assert config.total_batch_size % (config.batch_size * config.context_len) == 0, \
        "total_batch_size must be divisible by (batch_size * context_len)"

    print(f"tokens per iteration will be: {config.total_batch_size:,}")
    print(f"Total batch size = {config.total_batch_size}")
    print(f"grad_acum_steps = {config.grad_acum_steps}")

    # ── Data loading ──────────────────────────────────────────────────────────
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    val_loader = None
    test_loader = None

    if args.sharded:
        from prepare_data import DATASET_REGISTRY
        data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data")
        dataset_weights = {name: cfg["weight"] for name, cfg in DATASET_REGISTRY.items()}
        loader_args = dict(data_dir=data_dir, dataset_weights=dataset_weights,
                           batch_size=config.batch_size, context_len=config.context_len)
        data_loader = ShardedDataLoader(**loader_args, split="train")
        val_loader  = ShardedDataLoader(**loader_args, split="val")
        test_loader = ShardedDataLoader(**loader_args, split="test")
    else:
        BOOKS_DIR = os.path.join(PROJECT_ROOT, "..", "..", "..", "resources", "books")
        book_file = os.path.join(BOOKS_DIR, "tinyshakespeare.txt")
        with open(book_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Total len of text = {len(text)}")
        loader_args = dict(batch_size=config.batch_size, context_len=config.context_len)
        data_loader = DataLoader(text, **loader_args, split="train")
        val_loader  = DataLoader(text, **loader_args, split="val")
        test_loader = DataLoader(text, **loader_args, split="test")

    # ── Model setup ───────────────────────────────────────────────────────────
    mx.random.seed(1356)

    model = GPT2(config)
    mx.eval(model.parameters())

    # Optional bfloat16 — casts all float32 weights to bfloat16 for faster training
    # on Apple Silicon. MLX has no autocast; this is a global dtype change.
    if args.bfloat16:
        model.set_dtype(mx.bfloat16, predicate=lambda dtype: dtype == mx.float32)
        mx.eval(model.parameters())
        print("Model weights cast to bfloat16")

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = GPTTrainer(model, config, data_loader,
                         val_loader=val_loader, test_loader=test_loader,
                         checkpoint_interval=args.checkpoint_interval,
                         checkpoint_path=args.checkpoint_path)

    if args.resume:
        trainer.load_checkpoint()

    trainer.train_model(time_itr=True)


#   ── Run commands ────────────────────────────────────────────────────────────
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  DEFAULT MODE — single text file (tinyshakespeare), quick sanity test  │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   python train_gpt2_mlx.py
#   python train_gpt2_mlx.py --max_iters 20
#   python train_gpt2_mlx.py --bfloat16          # faster on Apple Silicon
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  SHARDED MODE — pre-tokenized datasets from prepare_data.py           │
#   │  (FineWeb-Edu 55% + Books 25% + TinyStories 20%)                     │
#   │                                                                       │
#   │  ⚠  Run prepare_data.py FIRST to create the shard files.             │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   # Step 1: Prepare shards (run once, see prepare_data.py for more options):
#   python prepare_data.py --max_tokens 10_000_000       # quick test
#   python prepare_data.py                                # full datasets
#
#   # Step 2: Train on sharded data:
#   python train_gpt2_mlx.py --sharded
#   python train_gpt2_mlx.py --sharded --max_iters 50
#   python train_gpt2_mlx.py --sharded --bfloat16
#
#   # Custom shard directory:
#   python train_gpt2_mlx.py --sharded --data_dir /path/to/shards
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  CHECKPOINTING & RESUME                                               │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   # Save checkpoint every 50 steps:
#   python train_gpt2_mlx.py --sharded --max_iters 500 --checkpoint_interval 50
#
#   # Custom checkpoint directory:
#   python train_gpt2_mlx.py --sharded --checkpoint_path models/my_run
#
#   # Resume training (loads from --checkpoint_path):
#   python train_gpt2_mlx.py --sharded --max_iters 500 --resume
#   python train_gpt2_mlx.py --sharded --max_iters 500 --resume --checkpoint_path models/my_run
#
#   # Checkpoint structure (directory, not a single file like PyTorch):
#   #   out/ckpt_train/
#   #     model.safetensors         (~496 MB for 124M params)
#   #     meta.json                 (step, val_loss, config)
#   #   out/ckpt_train_inference/
#   #     model.safetensors         (same, for inference — no optimizer state either way)
#   #     meta.json
#   #
#   # Note: MLX checkpoints don't save optimizer state (Adam momentum/variance).
#   # On resume, the optimizer starts fresh. This is standard MLX practice.
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  DIFFERENCES FROM PYTORCH VERSION (train_gpt2.py)                     │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   - No DDP / torchrun — MLX is single Apple Silicon device only
#   - No torch.compile — MLX uses mx.compile (not used here due to accumulation loop)
#   - No autocast — use --bfloat16 flag for global dtype cast instead
#   - Weight decay applies to ALL params (no param groups in MLX AdamW)
#   - Checkpoint: directory with safetensors + JSON (vs single .pt file)
#   - Optimizer state not checkpointed (restarts fresh on resume)
