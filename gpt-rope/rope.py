import torch


def build_rope_cache(seq_len, dim, device="cpu"):
    """
    Creates cosine and sine caches for RoPE.

    Args:
        seq_len: sequence length
        dim: head dimension (must be even)

    Returns:
        cos: [seq_len, dim/2]
        sin: [seq_len, dim/2]
    """

    assert dim % 2 == 0, "Dimension must be even"

    # k = [0,1,2,...]
    half_dim = dim // 2

    # Compute frequencies:
    #
    # omega_k = 1 / (10000^(2k/d))
    #
    freq_seq = torch.arange(half_dim, device=device)

    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    # Positions p = [0,1,2,...]
    positions = torch.arange(seq_len, device=device)

    # Compute angles:
    #
    # theta[p, k] = p * omega_k
    #
    theta = torch.outer(positions, inv_freq)

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    return cos, sin


def rotate_half(x):
    """
    Splits pairs and rotates:
    (x1,x2) -> (-x2,x1)

    Example:
    [a,b,c,d] -> [-b,a,-d,c]
    """

    # x is [B, CL, n_heads, head_dim]
    x1 = x[..., ::2] # [B, CL, n_heads, head_dim/2] in last dim take every 2nd ele start from 0
    x2 = x[..., 1::2] # [B, CL, n_heads, head_dim/2] in last dim take every 2nd ele start from 1

    rotated = torch.stack((-x2, x1), dim=-1) # [B, CL, n_heads, head_dim/2, 2]

    return rotated.flatten(-2) # [B, CL, n_heads, head_dim]


def apply_rope(x, cos, sin):
    """
    Applies RoPE to tensor x.

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim/2]
        sin: [seq_len, head_dim/2]
    """

    batch, seq_len, num_heads, head_dim = x.shape

    # Expand cos/sin so dimensions match x
    #
    # [seq_len, half_dim]
    # -> [1, seq_len, 1, half_dim]
    #
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Duplicate each value for dimension pairs
    #
    # [c1,c2] -> [c1,c1,c2,c2]
    #
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)

    # Apply rotation formula:
    #
    # x_rot = x*cos + rotate_half(x)*sin
    #
    x_rotated = (x * cos) + (rotate_half(x) * sin)

    return x_rotated


# ----------------------------------------------------
# Example usage
# ----------------------------------------------------

batch = 2
seq_len = 4
num_heads = 2
head_dim = 8

# Example Q tensor
q = torch.randn(batch, seq_len, num_heads, head_dim)

# Build RoPE cache
cos, sin = build_rope_cache(seq_len, head_dim)

# Apply RoPE
q_rope = apply_rope(q, cos, sin)

print("Original shape:", q.shape)
print("RoPE shape:", q_rope.shape)