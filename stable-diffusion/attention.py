import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embd: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embd, 3 * d_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (batch_size, seq_len, dim)
        input_shape = x.shape
        batch_size, seq_len, d_embd = input_shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # 3 tensors of (batch_size, seq_len, dim)
        q = q.view(interim_shape).transpose(1, 2) # (batch_size, h, seq_len, dim / h)
        k = k.view(interim_shape).transpose(1, 2) # (batch_size, h, seq_len, dim / h)
        v = v.view(interim_shape).transpose(1, 2) # (batch_size, h, seq_len, dim / h)
        weight = q @ k.transpose(-1, -2) # (batch_size, h, seq_len, seq_len)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v # (batch_size, h, seq_len, dim / h)
        output = output.transpose(1, 2) # (batch_size, seq_len, h, dim / h)
        output = output.reshape(input_shape) # (batch_size, seq_len, dim)
        output = self.out_proj(output)
        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embd: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embd, d_embd, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads

    def forward(self, x, y):
        # x: (latent): (batch_size, seq_len_q, dim_q)
        # y: (context): (batch_size, seq_len_kv, dim_kv)
        input_shape = x.shape
        batch_size, seq_len, d_embd = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).continuous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output