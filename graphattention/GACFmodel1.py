import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np

from graphattention.modules import MultiHeadAttention, ATTLayer
# GACFV1: 在原文的图卷积层之前加入attention。

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
        # attention
        features = self.att(userFeatures, itemFeatures)

        L1 = laplacianMat + selfLoop
        L1 = L1.cuda()
        feature1 = self.affine1(torch.sparse.mm(L1,features))

        L2 = laplacianMat.cuda()
        inter_feature = torch.mul(features,features)
        feature2 = self.affine2(torch.sparse.mm(L2, inter_feature))
        return feature1 + feature2

class GACFV1(Module):

    def __init__(self,userNum,itemNum,rt,embedSize=256,layers=[256,128,64], droprate=0.2, useCuda=True):

        super(GACFV1,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        
        self.GALayers = torch.nn.ModuleList()
        
                
        self.LaplacianMat = self.buildLaplacianMat(rt) # sparse format
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        self.transForm1 = nn.Linear(in_features=sum(layers)*2,out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GALayers.append(GALayer(From, To, 8, self.droprate))
        
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd.weight, std=0.01)
        nn.init.normal_(self.iEmbd.weight, std=0.01)

        
        nn.init.xavier_uniform_(self.transForm1.weight)
        self.transForm1.bias.data.zero_()

        nn.init.xavier_uniform_(self.transForm2.weight)
        self.transForm2.bias.data.zero_()

        nn.init.xavier_uniform_(self.transForm3.weight)
        self.transForm3.bias.data.zero_()

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
       
    def forward(self, userIdx, itemIdx):

        # gcf data propagation
        uidx, iidx, features = self.getFeatureMat()
        
        finalEmbd = features.clone()

        for ga in self.GALayers:

            userFeatures, itemFeatures = features[uidx], features[iidx]
            features = ga(userFeatures, itemFeatures, self.LaplacianMat, self.selfLoop)

            finalEmbd = torch.cat([finalEmbd, features.clone()],dim=1)

        itemIdx = itemIdx + self.userNum
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd,itemEmbd],dim=1)

        embd = nn.ReLU()(self.transForm1(embd))
        embd = nn.ReLU()(self.transForm2(embd))
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction        
        

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def buildLaplacianMat(self,rt):

        rt_item = rt['itemId'] + self.userNum
        uiMat = coo_matrix((rt['rating'], (rt['userId'], rt['itemId'])))

        uiMat_upperPart = coo_matrix((rt['rating'], (rt['userId'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.userNum+self.itemNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL