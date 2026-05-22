from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    context_len  = 256 # aka block_size, sequence_len, etc..
    batch_size = 64
    emb_dim = 384
    n_heads = 6
    n_transformer_blocks = 6
    dropout = 0.05
    vocab_size = 65

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        # Q,K,V all in a single matrix
        self.qkv = nn.Linear(config.emb_dim, 3 * config.emb_dim)
        # aka c_proj, projection
        self.w0 = nn.Linear(config.emb_dim, config.emb_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.manual = False
        if self.manual:
            self.register_buffer('tril', 
            torch.tril(torch.ones(config.context_len, config.context_len)).view(1,1,config.context_len,config.context_len) # We need to change the shape so that it works with (B, n_head, CL, ED)
            )
    
    def forward(self, x: torch.Tensor): # (B, CL, ED)
        B,CL,ED = x.shape
        
        # (B, CL, 3*ED)
        qkvx = self.qkv(x)

        # Split across 3rd dimension
        q,k,v = qkvx.split(ED, dim=2)

        # Now split as per n_heads
        head_dim = ED // self.config.n_heads
        q = q.view(B, CL, self.config.n_heads, head_dim) #(B, CL, n_head, head_dim)
        k = k.view(B, CL, self.config.n_heads, head_dim) #(B, CL, n_head, head_dim)
        v = v.view(B, CL, self.config.n_heads, head_dim) #(B, CL, n_head, head_dim)

        # Transpose now, transpose in not an inplcce operation, pytorch might move memory around
        q = q.transpose(1, 2) #(B, n_head, CL, head_dim)
        k = k.transpose(1, 2) #(B, n_head, CL, head_dim)
        v = v.transpose(1, 2) #(B, n_head, CL, head_dim)

        if self.manual:
            # q.kT
            attn = (q @ k.transpose(2,3)) * (head_dim ** -0.5) # (B, n_head, CL, CL)
            attn = attn.masked_fill(self.tril[:,:,:CL,:CL] == 0, float('-inf')) # (B, n_head, CL, CL)
            attn = F.softmax(attn, dim=-1) # (B, n_head, CL, CL)
            attn = self.attn_dropout(attn)
            attn_w = attn @ v # (B, n_head, CL, head_dim)
            attn_w = attn_w.transpose(1, 2) # (B, CL, n_head, head_dim)

        else:
            # Use PyTorch's built-in attention
            # dropout_p=0 during eval — SDPA doesn't check self.training unlike nn.Dropout
            dropout_p = self.config.dropout if self.training else 0.0
            attn_w = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            # SDPA returns (B, n_head, CL, head_dim) — transpose back before reshape
            attn_w = attn_w.transpose(1, 2) # (B, CL, n_head, head_dim)

        # We could do "attn_w = attn_w.contiguous().view(B, CL, ED)" as well, but need to call contiguous
        attn_w = attn_w.reshape(B, CL, ED) # (B, CL, ED)

        # output projection
        out = self.w0(attn_w) # (B, CL, ED)

        # Residual droupout
        out = self.resid_dropout(out)
        return out


class FFN(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.emb_dim, config.emb_dim * 4)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(config.emb_dim * 4, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x): # (B, CL, ED)
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x # (B, CL, ED)

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.emb_dim)
        self.attention = MultiHeadAttention(config)
        self.ln_ffn = nn.LayerNorm(config.emb_dim)
        self.ffn = FFN(config)
        
    def forward(self, x: torch.Tensor): # (B, CL, ED)
        x = x + self.attention(self.ln_attn(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x
        
        
class GPTModel2EX1(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.layer_dict = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.emb_dim),
            wpe = nn.Embedding(config.context_len, config.emb_dim),
            drop = nn.Dropout(config.dropout),
            transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_transformer_blocks)]),
            ln_final = nn.LayerNorm(config.emb_dim) 
        ))
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)
        self.lm_head.weight = nn.Parameter(self.layer_dict.wte.weight)

        # Init all Linear/Embedding weights to N(0, 0.02). See notes.md § Weight Initialization
        self.apply(self._init_weights)

        # Residual projections (w0, proj) get a smaller std to keep the residual stream stable at init.
        # std = 0.02 / sqrt(2 * n_layers). See notes.md § Weight Initialization → Scaled Residual Init
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

    def forward(self, inpTox, targetTox): # inp/trget shape - (B, CL)
        B, CL = inpTox.shape
        tok_emb = self.layer_dict.wte(inpTox) # (B, CL, ED)
        pos_emb = self.layer_dict.wpe(torch.arange(CL, device=inpTox.device)) # (CL, ED)
        x = tok_emb + pos_emb # (B, CL, ED)
        x = self.layer_dict.drop(x)

        for tb in self.layer_dict.transformer_blocks:
            x = tb(x)

        x = self.layer_dict.ln_final(x) # (B, CL, ED)
        if targetTox is not None:
            logits = self.lm_head(x) # (B, CL, V)
            B, CL, V = logits.shape
            logits = logits.view(B * CL, V)
            targetTox = targetTox.view(B * CL)
            loss = F.cross_entropy(logits, targetTox)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the context_len dim, without [-1] we would end up with. (B, ED) which can cause error. with [-1] we will have (B, 1, ED)
            loss = None
            
        return logits, loss
    
    def generate(self, inpTox: torch.Tensor, max_len: int, temperature=1.0, eot_token=None):
        for _ in range(max_len):
            # crop input to fit context length
            inpCtx = inpTox[:, -self.config.context_len:]

            # forward
            logits, _ = self(inpCtx, None)
            logits = logits[:, -1, :] / temperature # (B, ED)
            probs = F.softmax(logits, dim=-1)

            # get next token
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # stop if EOT token is generated
            if eot_token is not None and (next_token == eot_token).any():
                break

            # append to input
            inpTox = torch.cat([inpTox, next_token], dim=1)
        return inpTox