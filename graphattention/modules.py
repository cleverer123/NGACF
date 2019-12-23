import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
import torch.nn.functional as F

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

class ATTLayer_mask(Module):
    def __init__(self, dim, heads, dropout):
        super(ATTLayer_mask, self).__init__()
        self.att = MultiHeadAttention(heads, dim, dropout)
        
    def forward(self, features, mask):
        # residual = torch.cat([userFeatures,itemFeatures],dim=0)
        features = self.att(features, features, features, mask)
        # features = torch.cat([userFeatures, itemFeatures],dim=0)
        # features += residual
        return features

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(d_k)
    if mask is not None:
        # mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, -1e9)
        # scores = list(torch.chunk(scores, scores.shape[0], dim=0))
        # for idx, score in enumerate(scores):
        #     scores[idx] = score.squeeze().sparse_mask(mask).to_dense()
        # scores = torch.stack(scores)
    scores = torch.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

# def attention(q, k, v, d_k, mask=None, dropout=None):
    
#     scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(d_k)
    
#     if mask is not None:
#         # mask = mask.unsqueeze(1)
#         scores = scores.masked_fill(mask == 0, -1e9)
#     scores = torch.softmax(scores, dim=-1)

#     if dropout is not None:
#         scores = dropout(scores)
#     output = torch.matmul(scores, v)
#     return output

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
        
        self.__init_weight__()
    
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

# https://github.com/Diego999/pyGAT/blob/master/layers.py
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = torch.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
