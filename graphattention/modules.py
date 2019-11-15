import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np

class ATTLayer(Module):
    def __init__(self, dim, heads, dropout):
        super(ATTLayer, self).__init__()
        self.att_user = MultiHeadAttention(heads, dim, dropout)
        self.att_item = MultiHeadAttention(heads, dim, dropout)
        
    def forward(self, userFeatures, itemFeatures):
        # residual = torch.cat([userFeatures,itemFeatures],dim=0)
        userFeatures, itemFeatures = self.att_item(itemFeatures, userFeatures, userFeatures), self.att_user(userFeatures, itemFeatures, itemFeatures)
        features = torch.cat([userFeatures, itemFeatures],dim=0)
        # features += residual
        return features


def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = torch.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(Module):
    def __init__(self, heads, dim, dropout = 0.1):
        super().__init__()
        
        self.att = None
        
        self.dim = dim
        self.d_k = dim // heads
        self.h = heads
        
        self.q_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim)
    
    def __init_weight__(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.q_linear.bias.data.zero_()

        nn.init.xavier_uniform_(self.v_linear.weight)
        self.v_linear.bias.data.zero_()

        nn.init.xavier_uniform_(self.k_linear.weight)
        self.k_linear.bias.data.zero_() 

        nn.init.xavier_uniform_(self.out.weight)
        self.out.bias.data.zero_() 


    def forward(self, q, k, v, mask=None):        
        # perform linear operation and split into h heads
        # transpose to get dimensions  h * sl * dim
        q = self.q_linear(q).view(-1, self.h, self.d_k).transpose(0, 1)
        k = self.k_linear(k).view(-1, self.h, self.d_k).transpose(0, 1)
        v = self.v_linear(v).view(-1, self.h, self.d_k).transpose(0, 1)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(0,1).contiguous().view(-1, self.dim)       
        output = self.out(concat)
        return output

