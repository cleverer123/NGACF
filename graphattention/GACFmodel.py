import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np


# several models for recommendations

# RMSE
# SVD dim = 50 50 epoch RMSE = 0.931
# GNCF dim = 64 layer = [64,64,64] nn = [128,64,32,] 50 epoch RMSE = 0.916/RMSE =0.914
# NCF dim = 64 50 nn = [128,54,32] epoch 50 RMSE = 0.928

class SVD(Module):

    def __init__(self,userNum,itemNum,dim):
        super(SVD, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)
        self.uBias = nn.Embedding(userNum,1)
        self.iBias = nn.Embedding(itemNum,1)
        self.overAllBias = nn.Parameter(torch.Tensor([0]))

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        ubias = self.uBias(userIdx)
        ibias = self.iBias(itemIdx)

        biases = ubias + ibias + self.overAllBias
        prediction = torch.sum(torch.mul(uembd,iembd),dim=1) + biases.flatten()

        return prediction

class NCF(Module):

    def __init__(self,userNum,itemNum,dim,layers=[128,64,32,8]):
        super(NCF, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)
        self.fc_layers = torch.nn.ModuleList()
        self.finalLayer = torch.nn.Linear(layers[-1],1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.fc_layers.append(nn.Linear(From,To))

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        embd = torch.cat([uembd, iembd], dim=1)
        x = embd
        for l in self.fc_layers:
            x = l(x)
            x = nn.ReLU()(x)

        prediction = self.finalLayer(x)
        return prediction.flatten()

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
    
class ATTLayer(Module):
    def __init__(self, dim, heads, dropout):
        super(ATTLayer, self).__init__()
        self.att_user = MultiHeadAttention(heads, dim, dropout)
        self.att_item = MultiHeadAttention(heads, dim, dropout)
        
    def forward(self, userFeatures, itemFeatures):
        # residual = torch.cat([userFeatures,itemFeatures],dim=0)
        self.att_item(itemFeatures, userFeatures, userFeatures), self.att_user(userFeatures, itemFeatures, itemFeatures)
        features = torch.cat([userFeatures, itemFeatures],dim=0)
        # features += residual
        return features

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
        embd = self.transForm2(embd)
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

    
class GNNLayer(Module):

    def __init__(self,inF,outF):

        super(GNNLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF,out_features=outF)

    def forward(self, laplacianMat,selfLoop,features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # laplacianMat L = D^-1(A)D^-1 # 拉普拉斯矩阵
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat.cuda()
        L1 = L1.cuda()
        inter_feature = torch.mul(features,features)

        inter_part1 = self.linear(torch.sparse.mm(L1,features))
        inter_part2 = self.interActTransform(torch.sparse.mm(L2,inter_feature))

        return inter_part1+inter_part2


class GCF(Module):

    def __init__(self,userNum,itemNum,rt,embedSize=100,layers=[100,80,50],useCuda=True):

        super(GCF,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = self.buildLaplacianMat(rt) # sparse format
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        self.transForm1 = nn.Linear(in_features=sum(layers)*2, out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))

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

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def forward(self,userIdx,itemIdx):

        itemIdx = itemIdx + self.userNum
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)
        # gcf data propagation
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd,itemEmbd],dim=1)

        embd = nn.ReLU()(self.transForm1(embd))
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction
    

if __name__ == '__main__':
    from toyDataset.loaddata import load100KRatings

    rt = load100KRatings()
    userNum = rt['userId'].max()
    itemNum = rt['itemId'].max()

    rt['userId'] = rt['userId'] - 1
    rt['itemId'] = rt['itemId'] - 1
    # gcf = GCF(userNum,itemNum,rt)