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




block_size=8

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


class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx,targets=None):
        logits=self.token_embedding_table(idx)
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
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx.next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx.next),dim=1)
        return idx
    
model=BigramLanguageModel(vocab_size)
m=model.to(device)
optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
batch_size=32
for steps in range(20000):
    if steps%1000==0:
        losses=estimate_loss()
        print(losses)
    xb,yb=get_batch("train")

    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_token=500)[0].tolist()))