from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    # x: (..., head_dim) with interleaved pairs (q1, q2, q3, q4, ...)
    x1 = x[..., ::2]   # (..., head_dim/2) — first element of each pair
    x2 = x[..., 1::2]  # (..., head_dim/2) — second element of each pair
    # stack → (..., head_dim/2, 2), flatten last two dims → (..., head_dim)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # Frequencies per pair: [head_dim/2]
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Angles for each (position, pair): [max_seq_len, head_dim/2]
        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
        # Repeat so each pair shares its angle: [max_seq_len, head_dim]
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        self.register_buffer('cos_cache', angles.cos())
        self.register_buffer('sin_cache', angles.sin())

    def forward(self, seq_len: int):
        # Returns [1, 1, seq_len, head_dim] for broadcasting over (B, n_head, CL, head_dim)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin


@dataclass
class GPTConfig:
    context_len: int = 256
    batch_size: int = 64
    emb_dim: int = 384
    n_heads: int = 6
    n_transformer_blocks: int = 6
    dropout: float = 0.05
    vocab_size: int = 65


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.emb_dim, 3 * config.emb_dim)
        self.w0 = nn.Linear(config.emb_dim, config.emb_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.manual = False
        if self.manual:
            self.register_buffer('tril',
                torch.tril(torch.ones(config.context_len, config.context_len))
                    .view(1, 1, config.context_len, config.context_len)
            )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):  # (B, CL, ED)
        B, CL, ED = x.shape
        head_dim = ED // self.config.n_heads

        q, k, v = self.qkv(x).split(ED, dim=2)

        q = q.view(B, CL, self.config.n_heads, head_dim).transpose(1, 2)  # (B, n_head, CL, head_dim)
        k = k.view(B, CL, self.config.n_heads, head_dim).transpose(1, 2)
        v = v.view(B, CL, self.config.n_heads, head_dim).transpose(1, 2)

        # Apply RoPE to Q and K only — V carries content, not position
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        if self.manual:
            attn = (q @ k.transpose(2, 3)) * (head_dim ** -0.5)  # (B, n_head, CL, CL)
            attn = attn.masked_fill(self.tril[:, :, :CL, :CL] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            attn_w = attn @ v                   # (B, n_head, CL, head_dim)
            attn_w = attn_w.transpose(1, 2)     # (B, CL, n_head, head_dim)
        else:
            dropout_p = self.config.dropout if self.training else 0.0
            attn_w = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            attn_w = attn_w.transpose(1, 2)     # (B, CL, n_head, head_dim)

        attn_w = attn_w.reshape(B, CL, ED)
        out = self.resid_dropout(self.w0(attn_w))
        return out


class FFN(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.emb_dim, config.emb_dim * 4)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(config.emb_dim * 4, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.emb_dim)
        self.attention = MultiHeadAttention(config)
        self.ln_ffn = nn.LayerNorm(config.emb_dim)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attention(self.ln_attn(x), cos, sin)
        x = x + self.ffn(self.ln_ffn(x))
        return x


class GPTModelRoPE(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        head_dim = config.emb_dim // config.n_heads

        self.layer_dict = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.emb_dim),
            # wpe removed — position info is injected into Q/K via RoPE
            drop=nn.Dropout(config.dropout),
            transformer_blocks=nn.ModuleList([
                TransformerBlock(config) for _ in range(config.n_transformer_blocks)
            ]),
            ln_final=nn.LayerNorm(config.emb_dim)
        ))

        # One shared RoPE cache — same cos/sin applied across all layers and heads
        self.rope = RoPE(head_dim, config.context_len)

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

    def forward(self, inpTox, targetTox):  # (B, CL)
        B, CL = inpTox.shape
        x = self.layer_dict.wte(inpTox)   # (B, CL, ED) — no positional embedding added
        x = self.layer_dict.drop(x)

        cos, sin = self.rope(CL)           # [1, 1, CL, head_dim]
        for tb in self.layer_dict.transformer_blocks:
            x = tb(x, cos, sin)

        x = self.layer_dict.ln_final(x)

        if targetTox is not None:
            logits = self.lm_head(x)       # (B, CL, V)
            B, CL, V = logits.shape
            loss = F.cross_entropy(logits.view(B * CL, V), targetTox.view(B * CL))
        else:
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, V)
            loss = None

        return logits, loss

    def generate(self, inpTox: torch.Tensor, max_len: int, temperature=1.0, eot_token=None):
        for _ in range(max_len):
            inpCtx = inpTox[:, -self.config.context_len:]
            logits, _ = self(inpCtx, None)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            if eot_token is not None and (next_token == eot_token).any():
                break
            inpTox = torch.cat([inpTox, next_token], dim=1)
        return inpTox
