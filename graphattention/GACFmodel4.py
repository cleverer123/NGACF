import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np

from graphattention.modules import MultiHeadAttention, ATTLayer

# GACFV4: 在GACFV2的基础上，去掉attention作用于原始feature的部分，attention 只作用于interaction部分，结果做Element-wise乘积，形成interactive feature，然后Propagate

class GALayer(Module):
    def __init__(self, input_dim, out_dim, heads, dropout):
        super(GALayer, self).__init__()
        self.att = ATTLayer(input_dim, heads, dropout)
        
        self.affine1 = nn.Linear(input_dim, out_dim)
        self.affine2 = nn.Linear(input_dim, out_dim)
        
        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.affine1.weight)
        self.affine1.bias.data.zero_()

        nn.init.xavier_uniform_(self.affine2.weight)
        self.affine2.bias.data.zero_()
    
    def forward(self, userFeatures, itemFeatures, laplacianMat, selfLoop):
        
        features = torch.cat([userFeatures, itemFeatures],dim=0)
        L1 = laplacianMat + selfLoop
        L1 = L1.cuda()
        feature1 = nn.ReLU()(self.affine1(torch.sparse.mm(L1,features)))

        
        features = self.att(userFeatures, itemFeatures)
        L2 = laplacianMat.cuda()
        inter_feature = torch.mul(features,features)
        feature2 = nn.ReLU()(self.affine2(torch.sparse.mm(L2, inter_feature)))
        return feature1 + feature2

class GACFV4_layer(Module):
    def __init__(self,userNum,itemNum,adj,embedSize,layers, droprate, useCuda=True):
        super(GACFV4_layer, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        
        self.GALayers = torch.nn.ModuleList()
                    
        self.LaplacianMat = adj 
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GALayers.append(GALayer(From, To, 8, self.droprate))
        
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

    def forward(self, userIdx, itemIdx):

        # gcf data propagation
        uidx, iidx, features = self.getFeatureMat()
        
        finalEmbd = features.clone()

        for ga in self.GALayers:

            userFeatures, itemFeatures = features[uidx], features[iidx]
            features = ga(userFeatures, itemFeatures, self.LaplacianMat, self.selfLoop)

            finalEmbd = torch.cat([finalEmbd, features.clone()],dim=1)

        return finalEmbd        
        
class GACFV4(Module):
    def __init__(self, userNum,itemNum,adj,embedSize,layers, droprate):
        super(GACFV4, self).__init__()
        self.userNum = userNum
        self.gacf = GACFV4_layer(userNum,itemNum,adj,embedSize,layers, droprate)
    
    def forward(self, userIdx, itemIdx):
        gacf_embd = self.gacf(userIdx, itemIdx)

        itemIdx = itemIdx + self.userNum
        userEmbd = gacf_embd[userIdx]
        itemEmbd = gacf_embd[itemIdx]

        return torch.sum(userEmbd * itemEmbd, dim=1)
    

    