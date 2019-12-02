import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np


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


class GNNLayer(Module):

    def __init__(self,inF,outF):

        super(GNNLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF,out_features=outF)
        
        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

        nn.init.xavier_uniform_(self.interActTransform.weight)
        self.interActTransform.bias.data.zero_()

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


class GCF_BPR(Module):

    def __init__(self,userNum,itemNum,adj,embedSize=100,layers=[100,80,50],useCuda=True):

        super(GCF_BPR,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = adj
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        self.transForm1 = nn.Linear(in_features=sum(layers)*2, out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))
        
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

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def getFeatureMat(self):
        uidx = torch.LongTensor(np.arange(self.userNum))
        iidx = torch.LongTensor(np.arange(self.itemNum))
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()
        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    # def forward(self,userIdx,itemIdx):
    #     features = self.getFeatureMat()
    #     finalEmbd = features.clone()
    #     for gnn in self.GNNlayers:
    #         features = gnn(self.LaplacianMat,self.selfLoop,features)
    #         features = nn.ReLU()(features)
    #         finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

    #     self.finalEmbd = finalEmbd

    #     itemIdx = itemIdx + self.userNum
    #     userEmbd = self.finalEmbd[userIdx]
    #     itemEmbd = self.finalEmbd[itemIdx]
    #     if self.training:
    #         return torch.sum(userEmbd * itemEmbd, dim=1)
    #     else:
    #         return torch.mm(userEmbd, itemEmbd.transpose(1, 0)) 

    def forward(self,userIdx,itemIdx):
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        self.finalEmbd = finalEmbd
        if self.training:
            itemIdx = itemIdx + self.userNum
            userEmbd = self.finalEmbd[userIdx]
            itemEmbd = self.finalEmbd[itemIdx]
            
            embd = torch.cat([userEmbd,itemEmbd],dim=1)
            embd = nn.ReLU()(self.transForm1(embd))
            embd = nn.ReLU()(self.transForm2(embd))
            embd = self.transForm3(embd)
            prediction = embd.flatten()
        else: 
            itemIdx = itemIdx + self.userNum
            # userIdx = torch.tensor([1, 2]) -> userIdx = torch.tensor([1, 1, 1, 2, 2, 2])
            u_Idxs = userIdx.expand(itemIdx.shape[0], userIdx.shape[0]).transpose(1, 0).reshape(-1)
            # itemIdx = torch.tensor([3, 4, 5]) -> itemIdx = torch.tensor([3, 4, 5, 3, 4, 5])
            i_Idxs = itemIdx.expand(userIdx.shape[0], itemIdx.shape[0]).reshape(-1)

            userEmbd = self.finalEmbd[u_Idxs]
            itemEmbd = self.finalEmbd[i_Idxs]   

            embd = torch.cat([userEmbd,itemEmbd],dim=1)
            embd = nn.ReLU()(self.transForm1(embd))
            embd = nn.ReLU()(self.transForm2(embd))
            embd = self.transForm3(embd)
            prediction = embd.reshape(userIdx.shape[0], itemIdx.shape[0])
        return prediction

class GCF(Module):

    def __init__(self,userNum,itemNum,adj,embedSize=100,layers=[100,80,50],useCuda=True):

        super(GCF,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        # self.LaplacianMat = self.buildLaplacianMat(rt) # sparse format
        self.LaplacianMat = adj
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        self.transForm1 = nn.Linear(in_features=sum(layers)*2, out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))
        
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
        uidx = torch.LongTensor(np.arange(self.userNum))
        iidx = torch.LongTensor(np.arange(self.itemNum))
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def forward(self,userIdx,itemIdx):
        # if self.training:
            
            # userIdx = list(userIdx.cpu().data)
            # itemIdx = list(itemIdx.cpu().data)
            # gcf data propagation
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        self.finalEmbd = finalEmbd

        itemIdx = itemIdx + self.userNum
        userEmbd = self.finalEmbd[userIdx]
        itemEmbd = self.finalEmbd[itemIdx]
        
        embd = torch.cat([userEmbd,itemEmbd],dim=1)
        embd = nn.ReLU()(self.transForm1(embd))
        embd = nn.ReLU()(self.transForm2(embd))
        embd = self.transForm3(embd)
        prediction = embd.flatten()
        
        return prediction  