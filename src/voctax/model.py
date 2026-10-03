"""LLaMA-style model definition for vocab-tax.

Custom LLaMA-style decoder with GQA, RoPE, SwiGLU, RMSNorm.
Designed for small parameter counts (10M-50M non-embedding).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, seq_len):
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary(x, cos, sin):
    # x: (batch, heads, seq, dim)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int):
        super().__init__()
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.mlp = MLP(dim, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int,
                 n_heads: int, n_kv_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rotary = RotaryEmbedding(dim // n_heads, max_seq_len)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, dim * 4) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        s = input_ids.shape[1]
        cos, sin = self.rotary(self.embed.weight, s)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    def count_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel()
            total -= self.lm_head.weight.numel()
        return total
