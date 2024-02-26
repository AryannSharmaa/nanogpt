import torch
import torch.nn as nn 
from torch.nn import functional as F 
torch.manual_seed(1337)
device='cuda' if torch.cuda.is_available() else 'cpu'


with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()
chars=sorted(list(set(text)))
vocab_size=len(chars)
itos={i:ch for i,ch in enumerate(chars)}
stoi={ch:i for i,ch in itos.items()}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:''.join(itos[i] for i in l)

data=encode(text)
data=torch.tensor(data,dtype=torch.long)
n=int(0.9*len(data))
train=data[:n]
test=data[n:]

iters=5000
batch_size=32
block_size=8
n_embd=32

def get_batch(split):
    data=train if split=="train" else test 
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ["train","test"]:
        losses=torch.zeros(1000)
        for k in range(1000):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        wei=q@k.transpose(-2,-1) / C**0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        out=wei@v 
        return out 
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim=-1)

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU(n_embd),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
    def forward(self,x):
        x=self.sa(x)
        x=self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.pos_embedding_table=nn.Embedding(block_size,n_embd)
        self.sa_head=MultiHeadAttention(4,n_embd//4)
        self.ffwd=FeedForward(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_emb=self.token_embedding_table(idx)
        pos_emb=self.pos_embedding_table(torch.arange(T,device=device))
        x=tok_emb+pos_emb 
        x=self.sa_head(x)
        x=self.ffwd(x)
        logits=self.lm_head(x)

        if targets is None:
            loss=None 
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss 
    
    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx.next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx.next),dim=1)
        return idx
    
model=BigramLanguageModel()
m=model.to(device)
optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
for steps in range(iters):
    if steps%1000==0:
        losses=estimate_loss()
        print(losses)
    xb,yb=get_batch("train")

    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_token=500)[0].tolist()))