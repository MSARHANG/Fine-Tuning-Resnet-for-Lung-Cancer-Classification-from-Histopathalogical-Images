import torch 
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        
        assert embed_dim % num_heads == 0, 'embedding dimension should be deivisible by the number of attention heads'
        
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        if dropout != None :
            self.dropout = nn.Dropout(dropout)
        else :
            self.dropout = nn.Dropout(0)
            
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)
        
        self.out = nn.Linear(embed_dim,embed_dim)
        
    def forward(self, x):
        
        batch_size, num_patches, embed_dim = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.head_dim ** 0.5
        scores = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            scores = self.dropout(scores)
        
        attn_out = torch.matmul(scores,V)
        attn_out = attn_out.transpose(1,2).contiguous()
        attn_out = attn_out.view(batch_size, num_patches, embed_dim)
        output = self.out(attn_out)
    
        return scores, output
        

        
