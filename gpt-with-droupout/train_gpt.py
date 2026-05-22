import math
import torch
import contextlib
from dataclasses import dataclass
from gpt_model import GPTModel2EX1

@dataclass
class TrainConfig:
    learning_rate    = 3e-4
    min_lr           = 3e-5   # cosine decays to this (10% of max) — see get_lr()
    warmup_iters     = 100    # linear warmup before cosine decay starts
    max_train_iters  = 5000
    loss_eval_interval = 500
    loss_eval_itrs   = 200
    context_len      = 256    # aka block_size, sequence_len, etc..
    batch_size       = 64
    use_autocast     = True   # bfloat16 autocast — significant speedup on MPS/CUDA
    grad_clip        = 1.0    # gradient clipping max norm (0.0 = disabled)
    checkpoint_path  = None   # if set, saves checkpoint every loss_eval_interval iters

class TrainGPTModel:
    def __init__(self, model: GPTModel2EX1, config: TrainConfig, train_data, val_data, device):
        self.model = model
        self.config = config
        # AdamW decouples weight decay from gradient update — standard for GPT training
        self.optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        # Factory function — creates a fresh autocast context each call.
        # Reusing a single instance causes dtype mismatch when nested with torch.no_grad().
        # bfloat16 halves memory bandwidth per op with minimal precision loss.
        if config.use_autocast and device != 'cpu':
            self._autocast_ctx = lambda: torch.autocast(device_type=device, dtype=torch.bfloat16)
        else:
            self._autocast_ctx = contextlib.nullcontext

    def get_lr(self, itr):
        # Linear warmup for warmup_iters, then cosine decay to min_lr.
        # See notes.md § Cosine LR Schedule
        if itr < self.config.warmup_iters:
            return self.config.learning_rate * itr / self.config.warmup_iters
        if itr > self.config.max_train_iters:
            return self.config.min_lr
        decay_ratio = (itr - self.config.warmup_iters) / (self.config.max_train_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def train_model(self, start_iter=0):
        import time
        for i in range(start_iter, self.config.max_train_iters):
            # Update LR every iter — cosine schedule
            lr = self.get_lr(i)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr

            x, y = self.get_batch()
            self.optim.zero_grad()
            with self._autocast_ctx():
                _, loss = self.model(x, y)
            loss.backward()

            # Clip gradients — prevents rare large updates from destabilizing training
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optim.step()

            if (i % self.config.loss_eval_interval == 0):
                t_eval = time.time()
                out = self.estimate_loss()
                eval_ms = (time.time() - t_eval) * 1000
                print(f"Iter {i}: Train Loss: {out['train']:.4f}, Val Loss: {out['val']:.4f}  lr: {lr:.2e}  (eval: {eval_ms:.0f}ms)")

                if self.config.checkpoint_path:
                    self._save_checkpoint(i, out['val'])

    def _save_checkpoint(self, itr, val_loss):
        torch.save({
            'iter'                : itr,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'val_loss'            : val_loss,
        }, self.config.checkpoint_path)
        print(f"  checkpoint saved → {self.config.checkpoint_path}")

    def estimate_loss(self):
        # autocast must be the outer context to match the compilation context from training.
        # torch.no_grad() is inner — if @torch.no_grad() wraps the whole function (outer),
        # it conflicts with torch.compile's cached bfloat16 graph causing dtype mismatch.
        # Not calling model.eval() — toggling train/eval triggers torch.compile recompilation.
        out = {}
        for split in ['train', 'val']:
            # Keep losses on device — avoids MPS sync on every .item() call in the inner loop.
            # Single sync at losses.mean().item() at the end instead of loss_eval_itrs syncs.
            losses = torch.zeros(self.config.loss_eval_itrs, device=self.device)
            for k in range(self.config.loss_eval_itrs):
                x, y = self.get_batch(split)
                with self._autocast_ctx():
                    with torch.no_grad():
                        _, loss = self.model(x, y)
                losses[k] = loss  # no .item() — stays on device
            out[split] = losses.mean().item()  # single sync here
        return out

    def get_batch(self, split=None):
        data = self.val_data if split == 'val' else self.train_data
        random_batch_starts = torch.randint(len(data) - self.config.context_len, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.context_len] for i in random_batch_starts])
        y = torch.stack([data[i+1:i+self.config.context_len+1] for i in random_batch_starts])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
