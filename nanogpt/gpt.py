import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 256 # max context length
batch_size = 64 # number of sequences to process in parallel
max_iters = 5000 # training iterations
eval_interval = 500 # when to evaluate loss
eval_iters = 200 # 
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384 # embedding dimension
n_layers = 6 # transformer blocks
n_heads = 6 # attention heads
dropout = 0.2

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # unique chars in the text
vocab_size = len(chars)
stoi = {char: i for i, char in enumerate(chars)} # char to index mapping
itos = {i: char for i, char in enumerate(chars)} # index to char mapping
encode = lambda s: [stoi[c] for c in s] # string -> list of ints
decode = lambda l: [itos[i] for i in l] # list of ints -> string

data = torch.tensor(encode(text), dtype=torch.long) # convert text to tensor of ints
n = int(len(data) * 0.9) # data split
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate random starting indices
    x = torch.stack([data[i: i + block_size] for i in ix]) # generate input sequences
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]) # generate target sequences
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y) # forward pass through model
            losses[k] = loss.item() # convert single element loss tensor to float
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimension
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # causal mask to prevent attending to future tokens
        wei = F.softmax(wei, dim=-1) # get probabilities with softmax
        wei = self.dropout(wei) # regularize
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat outputs from all heads
        out = self.dropout(self.proj(out)) # restore embedding dimension and regularize
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # communication
        x = x + self.ffn(self.ln2(x)) # computation
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # maps token indices to embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # maps token positions to embeddings
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init_zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init_normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape # batch size, sequence length
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.lnf(x) # (B, T, n_embd)
        logits = self.head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None: # (B, T)
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # (B * T, C)
            targets = targets.view(B * T) # (B * T,)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            context = idx[:, -block_size:] # crop context to last block_size tokens if necessary
            logits, loss = self(context)
            logits = logits[:, -1, :] # focus on last token, (B, T, vocab_size) -> (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # sample next token with probability distribution, (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append next token, (B, T + 1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() # update params

context = torch.zeros((1, 1), dtype=torch.long, device=device) # [[0]]
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # output next 500 tokens