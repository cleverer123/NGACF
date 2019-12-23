import torch
import torch.nn as nn
import torch.nn.functional as F

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
        h = torch.mm(input, self.W) # ()
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

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training) # (batch_size, embedSize)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

# # 去掉GAT最后一层
# class MultiHeadSPGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, droprate, alpha, nheads):
#         super(MultiHeadGAT).__init__()

#         self.droprate = droprate

#         self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=droprate, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training) # (batch_size, embedSize)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         return x
        
class SPGACF(nn.Module):

    def __init__(self,userNum,itemNum,adj,embedSize,layers, droprate, useCuda=True):

        super(SPGACF, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        
        # self.ATTlayers = torch.nn.ModuleList()
        # self.GPlayers = torch.nn.ModuleList()
        # self.Affinelayers = torch.nn.ModuleList()
                
        # self.LaplacianMat = adj  
        # self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        # for From,To in zip(layers[:-1],layers[1:]):
        #     self.ATTlayers.append(ATTLayer_mask(From, 8, self.droprate))
        #     self.GPlayers.append(GPLayer())
        #     self.Affinelayers.append(nn.Linear(From,To))

        # self.gat = SpGAT_layer(nfeat=embedSize, 
        #         nhid=embedSize, 
        #         nclass=embedSize, 
        #         dropout=droprate, 
        #         nheads=8, 
        #         alpha=0.2)

        self.gat = SpGAT(nfeat=embedSize, # d_model
                nhid=8,                   # d_model / 8
                nclass=embedSize,         # d_model
                dropout=droprate, 
                nheads=8, 
                alpha=0.2)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd.weight, std=0.01)
        nn.init.normal_(self.iEmbd.weight, std=0.01)

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return uidx, iidx + self.userNum , features

    def forward(self, userIdx, itemIdx, mask):
        uidx, iidx, features = self.getFeatureMat() # features.shape: (batch_size, embedSize)
        
        features = self.gat(features, mask)

        itemIdx = itemIdx + self.userNum
        userEmbd = features[userIdx]
        itemEmbd = features[itemIdx]
        return torch.sum(userEmbd * itemEmbd, dim=1)

class MultiLayerSPGA(nn.Module):

    def __init__(self,userNum,itemNum,adj,embedSize,layers, droprate, useCuda=True):

        super(MultiLayerSPGA, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        
        self.GATlayers = torch.nn.ModuleList()
        self.GPlayers = torch.nn.ModuleList()
        self.Affinelayers = torch.nn.ModuleList()
                
        self.LaplacianMat = adj  
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        layers.insert(0, embedSize)
        for From,To in zip(layers[:-1], layers[1:]):
            self.GATlayers.append(SpGAT(nfeat=From, 
                nhid=8,                   
                nclass=From,         
                dropout=droprate, 
                nheads=8, 
                alpha=0.2))
            self.GPlayers.append(GPLayer())
            self.Affinelayers.append(nn.Linear(From,To))

        # self.gat = SpGAT(nfeat=embedSize, # d_model
        #         nhid=8,                   # d_model / 8
        #         nclass=embedSize,         # d_model
        #         dropout=droprate, 
        #         nheads=8, 
        #         alpha=0.2)
            
        # self.gp = GPLayer()

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd.weight, std=0.01)
        nn.init.normal_(self.iEmbd.weight, std=0.01)

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return uidx, iidx + self.userNum , features
    
    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def forward(self, userIdx, itemIdx, mask):
        uidx, iidx, features = self.getFeatureMat() # features.shape: (batch_size, embedSize)
        
        

        finalEmbd = features.clone()
        for gat, gp, aff in zip(self.GATlayers, self.GPlayers, self.Affinelayers):
            
            features = gat(features, mask)
            features = gp(features, self.LaplacianMat, self.selfLoop)           
            # features += residual

            features = nn.ReLU()(aff(features))

            finalEmbd = torch.cat([finalEmbd, features.clone()],dim=1)

        features = finalEmbd

        itemIdx = itemIdx + self.userNum
        userEmbd = features[userIdx]
        itemEmbd = features[itemIdx]
        return torch.sum(userEmbd * itemEmbd, dim=1)

class SPGAMGP(nn.Module):

    def __init__(self,userNum,itemNum,adj,embedSize,layers, droprate, useCuda=True):

        super(SPGAMGP, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        
        self.GPlayers = torch.nn.ModuleList()
        self.Affinelayers = torch.nn.ModuleList()
                
        self.LaplacianMat = adj  
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        self.gat = SpGAT(nfeat=embedSize, 
                nhid=8,                   
                nclass=embedSize,         
                dropout=droprate, 
                nheads=8, 
                alpha=0.2)

        layers.insert(0, embedSize)
        for From,To in zip(layers[:-1], layers[1:]):
            self.GPlayers.append(GPLayer())
            self.Affinelayers.append(nn.Linear(From,To))

        # self.gat = SpGAT(nfeat=embedSize, # d_model
        #         nhid=8,                   # d_model / 8
        #         nclass=embedSize,         # d_model
        #         dropout=droprate, 
        #         nheads=8, 
        #         alpha=0.2)
            
        # self.gp = GPLayer()

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd.weight, std=0.01)
        nn.init.normal_(self.iEmbd.weight, std=0.01)

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return uidx, iidx + self.userNum , features
    
    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def forward(self, userIdx, itemIdx, mask):
        uidx, iidx, features = self.getFeatureMat() # features.shape: (batch_size, embedSize)
        
        features = self.gat(features, mask)

        finalEmbd = features.clone()
        for gp, aff in zip(self.GPlayers, self.Affinelayers):
            
            features = gp(features, self.LaplacianMat, self.selfLoop)           
            features = nn.ReLU()(aff(features))

            finalEmbd = torch.cat([finalEmbd, features.clone()],dim=1)

        features = finalEmbd

        itemIdx = itemIdx + self.userNum
        userEmbd = features[userIdx]
        itemEmbd = features[itemIdx]
        return torch.sum(userEmbd * itemEmbd, dim=1)

class GPLayer(nn.Module):
    def __init__(self):
        super(GPLayer, self).__init__()
    
    def forward(self, features, laplacianMat, selfLoop):
        # residual = features
        L1 = laplacianMat + selfLoop
        L1 = L1.cuda()
        features = torch.sparse.mm(L1,features)
        # features += residual
        return features

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT. Code from https://github.com/Diego999/pyGAT/blob/master/layers.py"""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
    
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
        # print(torch.min(e_rowsum))

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