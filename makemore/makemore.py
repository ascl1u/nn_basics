import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size = None # length of input sequences
    vocab_size = None # input in range [0 ... vocab_size - 1]
    n_embd: int = 32
    n_embd2: int = 32

class Bigram(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))
    
    def get_block_size(self):
        return 1 # only use previous char to predict the next
    
    def forward(self, idx, targets=None):
        logits = self.logits[idx]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(self.vocab_size, config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size
    
    def forward(self, idx, targets=None):
        # idx shape: (batch_size, block_size)
        emb = self.wte(idx) # (batch_size, block_size, n_embd)
        x = emb.view(emb.size(0), -1) # flatten to (batch_size, block_size * n_embd)
        logits = self.mlp(x) # (batch_size, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets.view(-1)) # flatten to (batch_size, )
        return logits, loss