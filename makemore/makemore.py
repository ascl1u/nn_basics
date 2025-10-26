import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size = None # length of input sequences
    vocab_size = None # input in range [0 ... vocab_size - 1]

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