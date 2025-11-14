import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().init()
        self.token_embd = nn.Embedding(n_vocab, n_embd)
        self.position__embd = nn.Parameters(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        x = self.token_embd(tokens) # (batch_size, seq_len, dim)
        x += self.position_embd
        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        super().__init___()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function
        x = self.linear_2(x)
        x += residue
        return x

class CLIP(nn.Module):
    def __init__(self):
        self.embd = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embd(tokens) # (batch_size, seq_len, dim)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output