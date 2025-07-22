import torch
import torch.nn as nn
import torch.nn.functional as F 

class CasualSelfAttention(nn.Module):
    def __init__(self,n_embed,block_size,num_heads,dropout = 0.1):
        self.qkv = nn.Linear(n_embed,3*n_embed)
        self.dropout = nn.dropout(dropout)
        self.output_projection = nn.Linear(n_embed,n_embed)
        self.num_heads = num_heads
        self.block_size = block_size
       self.register_buffer("bias",torch.trill(torch.ones(self.block_size,self.block_size)).view(1,1,self.block_size,self.block_size))

    def forward(self,x):
        B,T,C = x.shape()
        q,k,v = self.qkv(x).split(dim = -1, n_embed)
        q = q.view(B,T,num_heads,n_embed/num_heads).transpose(1,2)
        k = k.view(B,T,num_heads,n_embed/num_heads).transpose(1,2)
        v = v.view(B,T,num_heads,n_embed/num_heads).transpose(1,2)
        score = q@k.transpose(-2,-1)/(C)**0.5
        score.masked_fill(self.bias[:,:,"T","T"] == 0, float("-inf"))
        score = F.softmax(score,dim = -1)
        y = score@v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        output = self.output_projection(y)
        return output

class MLP(nn.Module):
    def __init__(self,n_embed,hidden_dim,dropout = 0.1):
        self.fn1 = nn.Linear(n_embed,hidden_dim)
        self.fn2 = nn.Linear(hidden_dim,n_embed)
        self.dropout = nn.dropout(dropout)
    def forward(self,x):
        x = self.fn1(x)
        x = nn.GeLu(x)
        x = self.fn2(x)
        x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self,n_embed,block_size,num_heads,hidden_dim,dropout = 0.1):
        self.attention = CasualSelfAttention(n_embed,block_size,num_heads,dropout)
        self.mlp = MLP(n_embed,hidden_dim,dropout)
        self.dropout = nn.dropout(dropout)
        self.layernorm = nn.LayerNorm()
    def forward(self,x):
        x = x + self.attention(self.layernorm(x))
        x = x + self.mlp(self.layernorm(x))
        return x


class GPT2(nn.Module):
    def __init__(self,n_blocks,vocab_size,n_embed,block_size,num_heads,hidden_dim,dropout = 0.1):
        self.dropout = nn.dropout(0.1)
        self.lm_head = nn.Linear(n_embed,vocab_size)
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embeddings(vocab_size,n_embed)
                wpe = nn.Embeddings(block_size,n_embed)
                blocks = nn.ModuleList(AttentionBlock(n_embed,block_size,num_heads,hidden_dim,dropout) for _ in range(n_blocks))
                ln_f = nn.LayerNorm()
            )
        )
    def forward(self,idx):
        B,T = idx.size()
        l = torch.arange(0,T)
        token = self.transformer.wte(idx)
        pos = self.transformer.wpe(idx)
        x = self.dropout(token + pos)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(:,[-1],:)
        return logits




