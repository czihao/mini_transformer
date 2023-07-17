import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

## Hyper parameters
num_batches = 4
batch_size = 8
val_iter = 300
max_iter = 10000
n_embd = 32
head_size = n_embd
num_heads = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('./input.txt', 'r', encoding='utf-8') as f:
    shkspr = np.array(list(f.read()))

vocab = sorted(set(shkspr.tolist()))
vocab_size = len(vocab)

stoi = {ch:ind for ind, ch in enumerate(vocab)}
itos = {ind:ch for ind, ch in enumerate(vocab)}

# de/encode flattened N*B ndarray
encode = lambda batches : np.array([stoi[elmt] for elmt in batches])
decode = lambda batches : np.array([itos[elmt] for elmt in batches])

# TODO Split train val
def tr_val_split(data, p_train = 0.9):
    return data[0:int(p_train*data.size)], data[int(p_train*data.size)+1:]
x_tr, x_val = tr_val_split(shkspr)

def get_batch(split):
    data = x_tr if split == 'train' else x_val
    ind = torch.randint(0, data.size - batch_size - 1, (num_batches, 1), dtype=torch.int32).numpy()
    ind = np.concatenate([ind+i for i in range(batch_size)], axis=1)
    # print(data[ind])
    x = data[ind]
    y = data[ind+1]
    
    ## Uncomment for sanity check
    # for j in range(num_batches):
    #     for i in range(batch_size):
    #         x_str = ''.join(x[j, 0:i].tolist())
    #         print(f'x: {x_str}\ty: {y[j, i]}')
    # assert np.array_equal(x.flatten().reshape(num_batches, batch_size), x)
    return torch.tensor(encode(x.flatten()).reshape(num_batches, batch_size), dtype=torch.long).to(device),\
           torch.tensor(encode(y.flatten()).reshape(num_batches, batch_size), dtype=torch.long).to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(val_iter)
        for i in range(val_iter):
            x_b, y_b = get_batch(split)
            _, loss = model(x_b, y_b)
            losses[i] = loss.item()
        out[split] = losses.mean()
    return out

class AttnHead(nn.Module):
    def __init__(self, head_size=head_size, num_heads=num_heads):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask', torch.triu(torch.ones((batch_size, batch_size), dtype=torch.bool).reshape((1, 1, batch_size, batch_size)), diagonal=1))
    
    def forward(self, token):
        N, B, R = token.shape
        key = self.key(token).reshape(N, B, num_heads, head_size//num_heads).transpose(-2, -3)
        query = self.query(token).reshape(N, B, num_heads, head_size//num_heads).transpose(-2, -3)
        value = self.value(token).reshape(N, B, num_heads, head_size//num_heads).transpose(-2, -3)
        w = key @ query.transpose(-2, -1) * head_size**-.5
        w = w.masked_fill(self.mask[:,:,:B, :B], float('-inf'))
        w = F.softmax(w, dim=-1, dtype=torch.float32)
        out = w @ value
        out = out.transpose(1, 2).reshape(N, B, R)
        return out

class Zelda(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_representation_table = nn.Embedding(vocab_size, n_embd)
        self.position_representation_table = nn.Embedding(batch_size, n_embd)
        self.sa_head = AttnHead(head_size=head_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x_b, target = None):
        N, B = x_b.shape
        tok_emb = self.token_representation_table(x_b)
        pos_emb = self.position_representation_table(torch.arange(B, device=device))
        attn_tok = self.sa_head(pos_emb + tok_emb)
        logits = self.lm_head(attn_tok)
        
        N, B, R = logits.shape
        if target == None:
            loss = None
        else:
            logits = logits.view(N*B, R)
            target = target.view(N*B)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, context, max_tokens):
        for _ in range(max_tokens):
            context_ = context[:, -batch_size:]
            logits, _ = self(context_)
            # get most recent context
            recent = logits[:, -1, :]
            pdf = F.softmax(recent, dim=-1, dtype=torch.float32)
            y_pred = torch.multinomial(pdf, num_samples=1, replacement=True)
            context = torch.concatenate((context, y_pred), dim=-1)
        return context


x_b, y_b = get_batch('train')
# x_b = torch.tensor(x_b.reshape(N, B), dtype=torch.long)
# y_b = torch.tensor(y_b.reshape(N, B), dtype=torch.long)

zelda = Zelda()
zelda = zelda.to(device)
logits, loss = zelda(x_b, y_b)
# init_context = torch.zeros((1, 1), dtype=torch.long)
print(''.join(decode(zelda.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 100)[0].cpu().numpy()).tolist()))
print(f'Before train: {loss}')

adam = torch.optim.AdamW(zelda.parameters(), lr=1e-3)
for iter in range(max_iter):
    if iter % 1000 == 0:
        losses = estimate_loss(zelda)
        print(f'Train loss: {losses["train"]}, Val loss: {losses["val"]}')
    x_b, y_b = get_batch('train')
    # x_b = torch.tensor(x_b.reshape(N, B), dtype=torch.long)
    # y_b = torch.tensor(y_b.reshape(N, B), dtype=torch.long)
    logits, loss = zelda(x_b, y_b)
    adam.zero_grad(set_to_none=True)
    loss.backward()
    adam.step()
# print(f'After train: {loss}')
print(''.join(decode(zelda.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 100)[0].cpu().numpy()).tolist()))

