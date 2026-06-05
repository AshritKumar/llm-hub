import os
import math
import time
import inspect
import argparse
import torch
from dataclasses import dataclass
from torch.distributed import init_process_group, destroy_process_group
from gpt2_model import GPT2, GPTConfig, DataLoader, ShardedDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@dataclass
class DDPInfo:
    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    device: str
    master_process: bool


class GPTTrainer:
    def __init__(self, model: GPT2, config: GPTConfig, dataLoader: DataLoader, device,
                 val_loader=None, test_loader=None, eval_interval: int = 100, eval_batches: int = 20,
                 checkpoint_interval: int = 100, checkpoint_path: str = "out/ckpt_train.pt"):
        self.model = model
        self.config = config
        self.dataLoader = dataLoader
        self.device = device
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.start_step = 0
        self.optim = self.get_optimizer()

    @staticmethod
    def setup_ddp(config: GPTConfig):
        # various inits, derived attributes, I/O setup
        ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        out_dir = 'out'

        if ddp:
            backend = 'nccl'
            init_process_group(backend=backend)
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
            seed_offset = ddp_rank # each process gets a different seed
        else:
            # if not ddp, we are running on a single gpu, and one process
            ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
            master_process = True
            seed_offset = 0
            device = get_device()

        tokens_per_iter = config.grad_acum_steps * ddp_world_size * config.batch_size * config.context_len
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        if master_process:
            os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(1337 + seed_offset)
        return DDPInfo(ddp=ddp, rank=ddp_rank, local_rank=ddp_local_rank,
                       world_size=ddp_world_size, device=device, master_process=master_process)

    def get_optimizer(self):
        # params which require grad only
        parm_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Only decay weight matrices (dim >= 2) — not biases or LayerNorm scale/shift (dim < 2)
        decay_parms   = [p for _, p in parm_dict.items() if p.dim() >= 2]
        nodecay_parms = [p for _, p in parm_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_parms,   "weight_decay": self.config.weight_decay},
            {"params": nodecay_parms, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_parms)
        num_non_decay_params = sum(p.numel() for p in nodecay_parms)
        print(f"Number of decay params = {len(decay_parms)} params with sum {num_decay_params}, no. of non decay params = {len(nodecay_parms)} params with sum {num_non_decay_params}")

        # Use fused AdamW if available — faster on CUDA, not available on MPS
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in self.device

        return torch.optim.AdamW(optim_groups, self.config.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    def save_checkpoint(self, step, val_loss=None, inference_only=False):
        """Save a checkpoint to self.checkpoint_path.

        Two modes:
          inference_only=False (default): full training checkpoint (~1.5 GB for 124M params).
            Includes model weights (float32), optimizer state, config, step.
            Overwrites the same file each time (use checkpoint_path to control location).

          inference_only=True: lightweight inference checkpoint (~248 MB for 124M params).
            Saves model weights cast to bfloat16, no optimizer state.
            Saved alongside the training checkpoint with '_inference' suffix.
        """
        os.makedirs(os.path.dirname(self.checkpoint_path) or '.', exist_ok=True)

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        if inference_only:
            base, ext = os.path.splitext(self.checkpoint_path)
            path = f"{base}_inference{ext}"
            checkpoint = {
                'step': step,
                'model_state_dict': {k: v.cpu().to(torch.bfloat16) for k, v in raw_model.state_dict().items()},
                'gpt_config': self.config,
                'val_loss': val_loss,
                'dtype': 'bfloat16',
            }
        else:
            path = self.checkpoint_path
            checkpoint = {
                'step': step,
                'model_state_dict': {k: v.cpu() for k, v in raw_model.state_dict().items()},
                'optimizer_state_dict': self.optim.state_dict(),
                'gpt_config': self.config,
                'val_loss': val_loss,
            }

        torch.save(checkpoint, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Checkpoint saved: {path}  ({size_mb:.1f} MB)")

    def load_checkpoint(self, path=None):
        """Resume training from a checkpoint file.

        Args:
            path: Checkpoint file to load. Defaults to self.checkpoint_path.

        Handles both full training checkpoints and inference-only checkpoints.
        If loading an inference-only checkpoint (bfloat16, no optimizer), weights are
        cast back to float32 and optimizer state is reset (fresh Adam momentum).
        """
        path = path or self.checkpoint_path
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        state_dict = checkpoint['model_state_dict']
        if checkpoint.get('dtype') == 'bfloat16':
            state_dict = {k: v.float() for k, v in state_dict.items()}
            print("Loaded bfloat16 checkpoint — cast weights back to float32")

        raw_model.load_state_dict(state_dict)

        if 'optimizer_state_dict' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("No optimizer state in checkpoint — optimizer starts fresh")

        self.start_step = checkpoint['step'] + 1
        self.config = checkpoint['gpt_config']

        print(f"Resumed from {path} — starting at step {self.start_step}")

    def get_lr(self, itr):
        # Linear warmup for warmup_iters, then cosine decay to min_lr.
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
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = eval_loader.get_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                total_loss += loss.item()
        self.model.train()
        return total_loss / n_batches

    def train_model(self, n_itrs: int=None, use_ddp=False, master_process=False, world_size=1, time_itr=False):
        if self.config.max_train_iters is None:
            self.config.max_train_iters = n_itrs

        self.model.train()
        itrs = self.config.max_train_iters
        for step in range(self.start_step, itrs):
            acc_loss = 0.0
            t0 = time.time()

            # ── Validation check (every eval_interval steps + step 0 for baseline) ──
            if self.val_loader and step % self.eval_interval == 0:
                val_loss = self.evaluate(self.val_loader)
                if master_process:
                    print(f"Step {step}: val_loss = {val_loss:.4f}")

            # zero_grad once per outer step — gradients accumulate across all micro steps below
            self.optim.zero_grad()

            for micro_step in range(self.config.grad_acum_steps):
                x, y = self.dataLoader.get_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.split(':')[0], dtype=torch.bfloat16):
                    _, loss = self.model(x, y)

                # cross_entropy returns the mean loss over the micro-batch.
                # Summing raw micro-batch losses would give gradients grad_acum_steps× too large.
                # Dividing each by grad_acum_steps makes the accumulated gradient equivalent
                # to a single forward pass over the full (micro × grad_acum_steps) batch.
                loss = loss / self.config.grad_acum_steps
                acc_loss += loss.detach()
                if use_ddp:
                    # Suppress gradient sync on all micro steps except the last.
                    # DDP normally syncs gradients after every backward — suppressing it
                    # for intermediate steps avoids redundant all-reduce calls and cuts communication overhead.
                    self.model.require_backward_grad_sync = (micro_step == self.config.grad_acum_steps - 1)
                loss.backward()

            # Average the accumulated loss across all DDP ranks so the logged value matches the true mean loss.
            if use_ddp:
                dist.all_reduce(acc_loss, op=dist.ReduceOp.AVG)

            # Prevents gradient explosion by rescaling the global gradient vector if its L2 norm exceeds 1.0.
            # NaN-safe only for overflow (finite large grads), not for actual NaN gradients.
            # See NOTES.md for explanation of L2 norm and gradient clipping.
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update LR according to cosine schedule before each optimizer step
            lr_s = self.get_lr(step)
            for parm_grp in self.optim.param_groups:
                parm_grp['lr'] = lr_s

            self.optim.step()

            if step < 10:
                dt, token_throughput = 0.0, 0.0
                if time_itr:
                    # Block until the device finishes all pending ops so the wall-clock time is accurate.
                    dev = self.device.split(':')[0]
                    if dev == 'cuda':
                        torch.cuda.synchronize()
                    elif dev == 'mps':
                        torch.mps.synchronize()
                    t1 = time.time()
                    dt = (t1 - t0) * 1000

                    # Total tokens processed across all processes this step (B * CL * grad_acum_steps * world_size)
                    tokens_processed = self.config.batch_size * self.config.context_len * self.config.grad_acum_steps * world_size
                    token_throughput = tokens_processed / (t1 - t0)
                if master_process:
                    print(f"Step {step},  loss = {acc_loss.item():.4f} | dt = {dt:.2f} ms | norm = {norm:.4f} | LR = {lr_s:.2e} | token_throughput = {token_throughput:.2f} tokens/s")

            # ── Checkpoint (every checkpoint_interval steps, master process only) ──
            if master_process and self.checkpoint_interval and (step + 1) % self.checkpoint_interval == 0:
                ckpt_val_loss = self.evaluate(self.val_loader) if self.val_loader else None
                self.save_checkpoint(step, val_loss=ckpt_val_loss)

        # ── Final test evaluation ─────────────────────────────────────────────
        if self.test_loader:
            test_loss = self.evaluate(self.test_loader)
            if master_process:
                print(f"\nTest loss = {test_loss:.4f}")

        # Save final checkpoints — full (for resume) + inference-only (lightweight)
        if master_process:
            final_val_loss = self.evaluate(self.val_loader) if self.val_loader else None
            self.save_checkpoint(itrs - 1, val_loss=final_val_loss)
            self.save_checkpoint(itrs - 1, val_loss=final_val_loss, inference_only=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train GPT-2 from scratch")
    parser.add_argument("--sharded", action="store_true",
                        help="Use ShardedDataLoader (pre-tokenized .npy shards from prepare_data.py) "
                             "instead of the default single-file DataLoader")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to shard directory (default: <script_dir>/data). Only used with --sharded")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Number of training iterations (default: 10)")
    parser.add_argument("--checkpoint_path", type=str, default="out/ckpt_train.pt",
                        help="Path for saving/loading training checkpoint (default: out/ckpt_train.pt)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from --checkpoint_path")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save a checkpoint every N steps (default: 100)")
    args = parser.parse_args()

    MAX_ITRS = args.max_iters

    config = GPTConfig(
        context_len=1024,
        batch_size=8,
        emb_dim=768,
        n_heads=12,
        n_transformer_blocks=12,
        dropout=0.05,
        vocab_size=50304, # padded to nearest power of 2 for efficiency
        use_flash_attn=False,
        warmup_iters=10,
        learning_rate=6e-4,
        max_train_iters=MAX_ITRS,
        min_lr=6e-5,        # 10% of max LR
        weight_decay=0.1,
        total_batch_size=524288, # 2^19 — for multi-GPU CUDA; override below for single-device runs
    )
    assert config.total_batch_size % (config.batch_size * config.context_len) == 0, "Make sure total batch size is divisiible by (B*CL)"

    dpp_info = GPTTrainer.setup_ddp(config=config)

    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert config.total_batch_size % (config.batch_size * config.context_len * dpp_info.world_size) == 0, "Make sure total batch size is divisiible by (B*CL* world_size)"

    # We are dividing(reducing) the no of micro steps to take, since in DDP we will spin up `ddp_world_size` no. of processes
    config.grad_acum_steps = config.total_batch_size // (config.batch_size * config.context_len * dpp_info.world_size)

    device = dpp_info.device
    if dpp_info.master_process:
        print(f"Total batch size = {config.total_batch_size}")
        print(f"grad_acum_steps = {config.grad_acum_steps}")
        print(f"Using device: {device}")

    # ── Data loading ──────────────────────────────────────────────────────────
    # Two modes:
    #   Default:   reads a single text file (tinyshakespeare), tokenizes in memory.
    #              Good for quick local tests.
    #   --sharded: reads pre-tokenized .npy shards produced by prepare_data.py.
    #              Supports multiple datasets mixed by weight (55% FineWeb / 25% books / 20% TinyStories).
    #              Required for serious training runs with large datasets.
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    val_loader = None
    test_loader = None

    if args.sharded:
        from prepare_data import DATASET_REGISTRY
        data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data")
        dataset_weights = {name: cfg["weight"] for name, cfg in DATASET_REGISTRY.items()}
        loader_args = dict(data_dir=data_dir, dataset_weights=dataset_weights,
                           batch_size=config.batch_size, context_len=config.context_len,
                           num_processes=dpp_info.world_size, process_rank=dpp_info.rank)
        data_loader = ShardedDataLoader(**loader_args, split="train")
        val_loader  = ShardedDataLoader(**loader_args, split="val")
        test_loader = ShardedDataLoader(**loader_args, split="test")
    else:
        BOOKS_DIR = os.path.join(PROJECT_ROOT, "..", "..", "..", "resources", "books")
        book_file = os.path.join(BOOKS_DIR, "tinyshakespeare.txt")
        with open(book_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Total len of text = {len(text)}")
        loader_args = dict(batch_size=config.batch_size, context_len=config.context_len,
                           num_processes=dpp_info.world_size, process_rank=dpp_info.rank)
        data_loader = DataLoader(text, **loader_args, split="train")
        val_loader  = DataLoader(text, **loader_args, split="val")
        test_loader = DataLoader(text, **loader_args, split="test")

    # ── Model setup ───────────────────────────────────────────────────────────
    seed_offset = dpp_info.rank
    torch.manual_seed(1356 + seed_offset)

    model = GPT2(config)
    model.to(device)

    # Having issues in pytorch MPS with torch comple, uncomment when running on cuda GPU
    # model = torch.complie(model)

    # If we are using DDP we need to wrap the model in a DDP container so the comminucation and other things are handled by pytorch sommothly
    if dpp_info.ddp:
        model = DDP(model, device_ids=[dpp_info.local_rank])

    trainer = GPTTrainer(model, config, data_loader, device,
                         val_loader=val_loader, test_loader=test_loader,
                         checkpoint_interval=args.checkpoint_interval,
                         checkpoint_path=args.checkpoint_path)

    if args.resume:
        trainer.load_checkpoint()

    trainer.train_model(use_ddp=dpp_info.ddp, master_process=dpp_info.master_process, world_size=dpp_info.world_size, time_itr=True)

    if dpp_info.ddp:
        destroy_process_group()


#   ── Run commands ────────────────────────────────────────────────────────────
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  DEFAULT MODE — single text file (tinyshakespeare), quick sanity test  │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   # Basic run (10 iterations, tinyshakespeare):
#   python train_gpt2.py
#
#   # More iterations:
#   python train_gpt2.py --max_iters 20
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  SHARDED MODE — pre-tokenized datasets from prepare_data.py           │
#   │  (FineWeb-Edu 55% + Books 25% + TinyStories 20%)                     │
#   │                                                                       │
#   │  ⚠  Run prepare_data.py FIRST to create the shard files.             │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   # Step 1: Prepare shards (run once, see prepare_data.py for more options):
#   python prepare_data.py --max_tokens 10_000_000       # quick test (~10M tokens per dataset)
#   python prepare_data.py                                # full datasets (can take hours)
#
#   # Step 2: Train on sharded data:
#   python train_gpt2.py --sharded
#   python train_gpt2.py --sharded --max_iters 50
#
#   # Custom shard directory:
#   python train_gpt2.py --sharded --data_dir /path/to/shards
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  CHECKPOINTING & RESUME                                               │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   # Save checkpoint every 50 steps:
#   python train_gpt2.py --sharded --max_iters 500 --checkpoint_interval 50
#
#   # Custom checkpoint path:
#   python train_gpt2.py --sharded --checkpoint_path models/my_run.pt
#
#   # Resume training (loads from --checkpoint_path):
#   python train_gpt2.py --sharded --max_iters 500 --resume
#   python train_gpt2.py --sharded --max_iters 500 --resume --checkpoint_path models/my_run.pt
#
#   # Checkpoints saved to out/ckpt_train.pt by default (overwritten each save).
#   # A lightweight inference-only checkpoint is also saved at end of training:
#   #   out/ckpt_train.pt              (full, ~1.5 GB — for resuming)
#   #   out/ckpt_train_inference.pt    (bfloat16, ~248 MB — for inference)
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  MAC (MPS) — torchrun is NOT supported, NCCL requires CUDA            │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   python train_gpt2.py                                  # default mode
#   python train_gpt2.py --sharded                        # sharded mode
#   # Do NOT use torchrun on Mac — it will fail with NCCL/IPv6 errors.
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  MULTI-GPU with torchrun (CUDA only)                                  │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   # All GPUs on one machine (e.g. 4 GPUs):
#   torchrun --standalone --nproc_per_node=4 train_gpt2.py --sharded
#
#   # Across multiple machines (e.g. 2 nodes × 4 GPUs each):
#   # On node 0:
#   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<host> --master_port=29500 train_gpt2.py --sharded
#   # On node 1:
#   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=<host> --master_port=29500 train_gpt2.py --sharded
# python main/src/kar_nn/gpt2_replica/train_gpt2.py --sharded --max_iters 2 --checkpoint_path /Users/ashritkuma.samudrala/lnex/ex_llm_rag/main/resources/models/gpt2/gp2_model_check_point_train.model