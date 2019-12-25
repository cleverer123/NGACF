import torch
import torch.nn as nn
import torch.nn.functional as F

class SPUIGACF(nn.Module):

    def __init__(self,userNum,itemNum,adj,embedSize,layers, droprate, useCuda=True):

        super(SPUIGACF, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)

        self.gat = SpUIGAT(nfeat=embedSize, # d_model
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
        mask = mask.squeeze()
        uidx, iidx, features = self.getFeatureMat() # features.shape: (batch_size, embedSize)

        # finalEmbd = features.clone()
        features = self.gat(uidx, iidx, features, mask)
        # finalEmbd = torch.cat([finalEmbd, features.clone()],dim=1)

        itemIdx = itemIdx + self.userNum
        userEmbd = features[userIdx]
        itemEmbd = features[itemIdx]
        return torch.sum(userEmbd * itemEmbd, dim=1)

class SPUIGAGPCF(nn.Module):

    def __init__(self,userNum,itemNum,adj,embedSize,layers, droprate, useCuda=True):

        super(SPUIGAGPCF, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)

        self.gat = SpUIGAT(nfeat=embedSize, # d_model
                nhid=8,                   # d_model / 8
                nclass=embedSize,         # d_model
                dropout=droprate, 
                nheads=8, 
                alpha=0.2)

        self.GPlayers = torch.nn.ModuleList()
        self.Affinelayers = torch.nn.ModuleList()
        self.LaplacianMat = adj  
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        layers.insert(0, embedSize)
        for From,To in zip(layers[:-1], layers[1:]):
            self.GPlayers.append(GPLayer())
            self.Affinelayers.append(nn.Linear(From,To))

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
        mask = mask.squeeze()
        uidx, iidx, features = self.getFeatureMat() # features.shape: (batch_size, embedSize)
        
        features = self.gat(uidx, iidx, features, mask)
        
        finalEmbd = features.clone()

        for gp, aff in zip(self.GPlayers, self.Affinelayers):
            
            features = gp(features, self.LaplacianMat, self.selfLoop)           
            features = nn.ReLU()(aff(features))

            finalEmbd = torch.cat([finalEmbd, features.clone()],dim=1)

        itemIdx = itemIdx + self.userNum
        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        return torch.sum(userEmbd * itemEmbd, dim=1)

class GPLayer(nn.Module):
    def __init__(self):
        super(GPLayer, self).__init__()
    
    def forward(self, features, laplacianMat, selfLoop):
        # residual = features
        # L1 = laplacianMat + selfLoop
        L1 = laplacianMat
        L1 = L1.cuda()
        features = torch.sparse.mm(L1,features)
        # features += residual
        return features

class SpUIGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpUIGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpUIGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpUIGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, uidx, iidx, features, adj):
        features = F.dropout(features, self.dropout, training=self.training)
        
        # print('uidx.device', uidx.device, 'features.device', features.device, 'gat.device', next(self.attention_0.parameters()).device)
        
        features = torch.cat([att(uidx, iidx, features, adj) for att in self.attentions], dim=1)
        features = F.dropout(features, self.dropout, training=self.training)
        features = F.elu(self.out_att(uidx, iidx, features, adj))
        return features

class SpUIGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super(SpUIGraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat

        self.W_u = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_normal_(self.W_u.data, gain=1.414)
        self.W_i = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_normal_(self.W_i.data, gain=1.414)
        # self.W_u = nn.Linear(in_dim, out_dim)
        # self.W_i = nn.Linear(in_dim, out_dim)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.spmm = SpecialSpmm()

    def forward(self, uidx, iidx, features, adj):
        # dv = 'cuda' if input.is_cuda else 'cpu'
        dv = 'cuda'

        uidx = uidx.to(self.W_u.device)
        iidx = iidx.to(self.W_u.device)
        features = features.to(self.W_u.device)

        num_user = uidx.size()[0]
        num_item = iidx.size()[0]
        # adj: userid X item_id 
        edge = adj.nonzero().t() # edge: (2, num_nodes), this can be seen as sparse indexs
        
        # u_h = self.W_u(features[uidx])
        # i_h = self.W_i(features[iidx])
        u_h = torch.mm(features[uidx], self.W_u)  # (num_user, out_dim)
        i_h = torch.mm(features[iidx], self.W_i)  # (num_item, out_dim)

        edge_h = torch.cat((u_h[edge[0, :], :], i_h[edge[1, :], :]), dim=1).t()  # cat((num_nodes, out_dim), (num_nodes, out_dim), dim=1) -> (num_nodes, 2*out_dim)
        # attention coefficients. See paper 'Graph Attention Networks' Eq.1 and Eq.3 numerator (公式3中的分子)
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze())).to(edge.device) # edge_e: (num_nodes) , this can be seen as sparse values
        
        # See paper 'Graph Attention Networks' Eq.3 denominator  (公式3中的分母)
        
        a = torch.sparse_coo_tensor(edge, edge_e, torch.Size([num_user, num_item]))
        # e_rowsum = self.spmm(edge, edge_e, torch.Size([num_user, num_item]), torch.ones(size=(num_item, 1), device=dv)) # e_rowsum: (num_user)
        e_rowsum = torch.sparse.mm(a, torch.ones(size=(num_item, 1), device=dv))
        assert not (e_rowsum == 0).sum() # 如果中断，则有行和为0的情况，会出现除0

        e_colsum = torch.sparse.mm(a.t(), torch.ones(size=(num_user, 1), device=dv))
        # assert not (e_colsum == 0).sum() # 如果中断，则有行和为0的情况，会出现除0

        # 对于user-item，对行求和，再除以行和，那每一行表示，该行对应的用户，对于每一项item的注意力分布。

        edge_e = self.dropout(edge_e)
        
        a = torch.sparse_coo_tensor(edge, edge_e, torch.Size([num_user, num_item]))
        # feature sum of neighbours of user 参考GAT公式4
        # attentive_items = self.spmm(edge, edge_e, torch.Size([num_user, num_item]), i_h) # (num_user, out_dim)
        attentive_items = torch.sparse.mm(a.to(i_h.device), i_h)
        # print('attentive_items.device', attentive_items.device, 'e_rowsum.device', e_rowsum.device)
        attentive_items = attentive_items.div(e_rowsum.to(i_h.device))  # (num_user, out_dim)
        u_h_prime = u_h + attentive_items # (num_user, out_dim)
        assert not torch.isnan(u_h_prime).any()
        
        # feature sum of neighbours of items
        attentive_users = torch.sparse.mm(a.t().to(u_h.device), u_h)
        attentive_users = attentive_users.div(e_colsum.to(u_h.device)) # (num_item, out_dim)
        attentive_users[torch.isnan(attentive_users)] = 0.0
        i_h_prime = i_h + attentive_users
        assert not torch.isnan(i_h_prime).any()
        
        h_prime = torch.cat([u_h_prime,i_h_prime], dim=0)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

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