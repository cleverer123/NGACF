import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np

from graphattention.modules import MultiHeadAttention, ATTLayer

# several models for recommendations

# RMSE
# SVD dim = 50 50 epoch RMSE = 0.931
# GNCF dim = 64 layer = [64,64,64] nn = [128,64,32,] 50 epoch RMSE = 0.916/RMSE =0.914
# NCF dim = 64 50 nn = [128,54,32] epoch 50 RMSE = 0.928



# GraphPropagationLayer
class GPLayer(Module):
    def __init__(self):
        super(GPLayer, self).__init__()
    
    def forward(self, features, laplacianMat, selfLoop):
        # residual = features
        L1 = laplacianMat + selfLoop
        L1 = L1.cuda()
        features = torch.sparse.mm(L1,features)
        # features += residual
        return features

class GACF(Module):

    def __init__(self,userNum,itemNum,rt,embedSize=256,layers=[256,128,64], droprate=0.2, useCuda=True):

        super(GACF,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.droprate = droprate
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        
        self.ATTlayers = torch.nn.ModuleList()
        self.GPlayers = torch.nn.ModuleList()
        self.Affinelayers = torch.nn.ModuleList()
                
        self.LaplacianMat = self.buildLaplacianMat(rt) # sparse format
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        self.transForm1 = nn.Linear(in_features=sum(layers)*2,out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.ATTlayers.append(ATTLayer(From, 8, self.droprate))
            self.GPlayers.append(GPLayer())
            self.Affinelayers.append(nn.Linear(From,To))

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd.weight, std=0.01)
        nn.init.normal_(self.iEmbd.weight, std=0.01)

        for m in self.Affinelayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

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

        for att, gp, aff in zip(self.ATTlayers, self.GPlayers, self.Affinelayers):
            residual = features

            userFeatures, itemFeatures = features[uidx], features[iidx]
            features = att(userFeatures, itemFeatures)
            features = gp(features, self.LaplacianMat, self.selfLoop)
            
            features += residual

            features = nn.ReLU()(aff(features))

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



if __name__ == '__main__':
    from toyDataset.loaddata import load100KRatings

    rt = load100KRatings()
    userNum = rt['userId'].max()
    itemNum = rt['itemId'].max()

    rt['userId'] = rt['userId'] - 1
    rt['itemId'] = rt['itemId'] - 1
    # gcf = GCF(userNum,itemNum,rt)