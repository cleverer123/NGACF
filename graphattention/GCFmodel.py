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

class MF_layer(Module):
    def __init__(self, userNum, itemNum, dim):
        super(MF_layer, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)

    def forward(self, userIdx, itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        return uembd * iembd

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

class GCF(Module):
    def __init__(self,userNum,itemNum,adj,embedSize=100,layers=[100,80,50],useCuda=True):

        super(GCF,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = adj
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))
        
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd.weight, std=0.01)
        nn.init.normal_(self.iEmbd.weight, std=0.01)

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

    def forward(self,userIdx,itemIdx):
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

        return torch.sum(userEmbd * itemEmbd, dim=1)

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

class NGCF_layer(Module):
# Neural Graph collarative filtering layer
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCF_layer,self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = adj
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))
        
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd.weight, std=0.01)
        nn.init.normal_(self.iEmbd.weight, std=0.01)

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def getFeatureMat(self):
        uidx = torch.LongTensor(np.arange(self.userNum)).cuda()
        iidx = torch.LongTensor(np.arange(self.itemNum)).cuda()
        
        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def forward(self,userIdx,itemIdx):
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        # self.finalEmbd = finalEmbd
        return finalEmbd
        # itemIdx = itemIdx + self.userNum
        # userEmbd = self.finalEmbd[userIdx]
        # itemEmbd = self.finalEmbd[itemIdx]

        # return torch.sum(userEmbd * itemEmbd, dim=1)

class NGCFMF_layer(Module):
# Neural Graph collaborative filtering with matrix factorization output
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMF_layer,self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.ngcf = NGCF_layer(userNum,itemNum,adj,embedSize,layers)

    def forward(self,userIdx,itemIdx):        
        ngcf_embd = self.ngcf(userIdx,itemIdx)

        itemIdx = itemIdx + self.userNum
        userEmbd = ngcf_embd[userIdx]
        itemEmbd = ngcf_embd[itemIdx]

        return userEmbd * itemEmbd

class NGCFMF(Module):
# Neural Graph collaborative filtering with matrix factorization output
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMF,self).__init__()
        
        self.ngcfmf = NGCFMF_layer(userNum,itemNum,adj,embedSize,layers)

    def forward(self,userIdx,itemIdx):        
        ngcfmf_embd = self.ngcfmf(userIdx,itemIdx)
        # return torch.sum(userEmbd * itemEmbd, dim=1)   
        return torch.sum(ngcfmf_embd, dim=1)

class CFMLP_layer(Module):
# collaborative filtering with mlp output layer
    def __init__(self,embedSize,layers):

        super(CFMLP_layer,self).__init__()

        self.transForm1 = nn.Linear(in_features=sum(layers)*2, out_features=sum(layers))
        self.transForm2 = nn.Linear(in_features=sum(layers), out_features=embedSize)
        # self.transForm3 = nn.Linear(in_features=sum(layers)//2, out_features=embedSize)

        self._init_weight_()

    def _init_weight_(self):

        nn.init.xavier_uniform_(self.transForm1.weight)
        self.transForm1.bias.data.zero_()

        nn.init.xavier_uniform_(self.transForm2.weight)
        self.transForm2.bias.data.zero_()

        # nn.init.xavier_uniform_(self.transForm3.weight)
        # self.transForm3.bias.data.zero_()

    def forward(self,embd):       
        embd = nn.ReLU()(self.transForm1(embd))
        embd = nn.ReLU()(self.transForm2(embd))
        # embd = nn.ReLU()(self.transForm3(embd))
        return embd  

class NGCFMLP_layer(Module):
# Neural Graph collaborative filtering with mlp output layer
    def __init__(self,userNum,itemNum,adj,embedSize,layers):
        super(NGCFMLP_layer,self).__init__()
        self.userNum = userNum
        self.ngcf = NGCF_layer(userNum,itemNum,adj,embedSize,layers)
        self.cfmlp = CFMLP_layer(embedSize,layers)
        
    def forward(self,userIdx,itemIdx):
        ngcf_embd = self.ngcf(userIdx,itemIdx)
        itemIdx = itemIdx + self.userNum
        userEmbd = ngcf_embd[userIdx]
        itemEmbd = ngcf_embd[itemIdx]
       
        embd = torch.cat([userEmbd,itemEmbd],dim=1)
        
        return self.cfmlp(embd)

class NGCFMLP(Module):
# Neural Graph collaborative filtering with mlp output layer
    def __init__(self,userNum,itemNum,adj,embedSize,layers):
        super(NGCFMLP, self).__init__()
        self.ngcfmlp = NGCFMLP_layer(userNum,itemNum,adj,embedSize,layers)
        self.output = nn.Linear(embedSize, 1)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, userIdx, itemIdx):
        ngcfmlp_embd = self.ngcfmlp(userIdx, itemIdx)
        return self.output(ngcfmlp_embd).flatten()

    
class NGCFMFMLP_layer(Module):
# Neural Graph collaborative filtering with matrix factorization output concat with matrix factorization output
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMFMLP_layer,self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        
        self.ngcf = NGCF_layer(userNum,itemNum,adj,embedSize,layers)
        self.cfmlp = CFMLP_layer(embedSize,layers)
        self.output = nn.Linear(sum(layers) + embedSize, 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self,userIdx,itemIdx):
        ngcf_embd = self.ngcf(userIdx,itemIdx)
        itemIdx = itemIdx + self.userNum
        userEmbd = ngcf_embd[userIdx]
        itemEmbd = ngcf_embd[itemIdx]
        
        ngcfmf_output = userEmbd * itemEmbd
        cfmlp_output = torch.cat([userEmbd,itemEmbd],dim=1)
        cfmlp_output = self.cfmlp(cfmlp_output)
        concat = torch.cat((ngcfmf_output, cfmlp_output), -1)
        
        return concat

class NGCFMFMLP(Module):
    def __init__(self, userNum, itemNum, adj, embedSize, layers):
        super(NGCFMFMLP, self).__init__()
        self.ngcfmfmlp = NGCFMFMLP_layer(userNum, itemNum, adj, embedSize, layers)
        self.output = nn.Linear(sum(layers) + embedSize, 1)
    
    def forward(self, userIdx, itemIdx):
        ngcfmfmlp_embd = self.ngcfmfmlp(userIdx, itemIdx)
        return self.output(ngcfmfmlp_embd).flatten()

class NGCFMF_concat_MF(Module):
# Neural Graph collaborative filtering with matrix factorization output concat with matrix factorization
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMF_concat_MF,self).__init__()
        self.mf = MF_layer(userNum, itemNum, embedSize)
        self.ngcfmf = NGCFMF_layer(userNum,itemNum,adj,embedSize,layers)

        self.output = nn.Linear(sum(layers) + embedSize, 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()
    
    def forward(self,userIdx,itemIdx):
        mf_output = self.mf(userIdx, itemIdx)   # (, embedSize)
        gcf_output = self.ngcfmf(userIdx,itemIdx) # (, sum(layers))
        concat = torch.cat((mf_output, gcf_output), -1)
        
        return self.output(concat).flatten()

class MLP_layer(Module):
    def __init__(self,userNum,itemNum, embedSize):
        super(MLP_layer, self).__init__()
        self.uEmbd_mlp = nn.Embedding(userNum,embedSize)
        self.iEmbd_mlp = nn.Embedding(itemNum,embedSize)
        
        n_layers = 3
        mlp_layers = []
        mlp_layers.append(nn.Linear(embedSize * 2, embedSize * (2 ** n_layers)))
        mlp_layers.append(nn.ReLU(inplace=True))
        for i in range(n_layers):
            input_dim = embedSize * (2 ** (n_layers - i))
            mlp_layers.append(nn.Linear(input_dim, input_dim // 2))
            mlp_layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp_layers)

    def _init_weight_(self):
        nn.init.normal_(self.uEmbd_mlp.weight, std=0.01)
        nn.init.normal_(self.iEmbd_mlp.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, userIdx, itemIdx):
        user_embs_mlp = self.uEmbd_mlp(userIdx)
        item_embs_mlp = self.iEmbd_mlp(itemIdx)

        interaction = torch.cat([user_embs_mlp, item_embs_mlp], dim=-1)
        mlp_output = self.mlp(interaction)
        return mlp_output

class NGCFMF_concat_MLP(Module):
# Neural Graph collaborative filtering with matrix factorization output concat with MLP
    def __init__(self,userNum,itemNum,adj,embedSize,layers):
        super(NGCFMF_concat_MLP,self).__init__()
        
        self.ngcfmf = NGCFMF_layer(userNum,itemNum,adj,embedSize,layers)
        self.mlp = MLP_layer(userNum,itemNum, embedSize)

        self.output = nn.Linear(sum(layers) + embedSize, 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.output.weight)
        # nn.init.kaiming_uniform_(self.output.weight, a=1)
        self.output.bias.data.zero_()

    def forward(self, userIdx, itemIdx):

        mlp_output = self.mlp(userIdx, itemIdx)  # (, embedSize)

        ngcfmf_output = self.ngcfmf(userIdx, itemIdx) # (, sum(layers))

        concat = torch.cat((mlp_output, ngcfmf_output), -1)
        
        return self.output(concat).flatten()

class NGCFMF_concat_MF_MLP(Module):
# Neural Graph collaborative filtering with MF output concat with matrix factorization output and MLP output
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMF_concat_MF_MLP,self).__init__()
        
        self.ngcfmf = NGCFMF_layer(userNum, itemNum, adj, embedSize, layers)
        self.mf = MF_layer(userNum, itemNum, embedSize)
        self.mlp = MLP_layer(userNum, itemNum, embedSize)

        self.output = nn.Linear(sum(layers) + embedSize*2, 1)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self,userIdx,itemIdx):
        mf_output = self.mf(userIdx, itemIdx) # (, embedSize)

        mlp_output = self.mlp(userIdx, itemIdx)  # (, embedSize)

        ngcfmf_output = self.ngcfmf(userIdx, itemIdx) # (, sum(layers))

        concat = torch.cat((mf_output, mlp_output, ngcfmf_output), -1)
        
        return self.output(concat).flatten()

class NGCFMLP_concat_MF(Module):
# Neural Graph collaborative filtering with mlp output concat with matrix factorization
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMLP_concat_MF,self).__init__()
    
        self.ngcfmlp = NGCFMLP_layer(userNum,itemNum,adj,embedSize,layers)
        self.mf = MF_layer(userNum, itemNum, embedSize)
        self.output = nn.Linear(embedSize*2, 1)
        
        self._init_weight_()

    def _init_weight_(self):

        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self,userIdx,itemIdx):
        mf_output = self.mf(userIdx,itemIdx) # (, embedSize)

        ngcfmlp_output = self.ngcfmlp(userIdx, itemIdx) # (, embsize)

        concat = torch.cat((mf_output, ngcfmlp_output), -1)
        
        return self.output(concat).flatten()

class NGCFMLP_concat_MLP(Module):
# Neural Graph collaborative filtering with mlp output concat with mlp
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMLP_concat_MLP,self).__init__()
    
        self.ngcfmlp = NGCFMLP_layer(userNum,itemNum,adj,embedSize,layers)
        self.mlp = MLP_layer(userNum, itemNum, embedSize)
        self.output = nn.Linear(embedSize*2, 1)
        
        self._init_weight_()

    def _init_weight_(self):

        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self,userIdx,itemIdx):

        ngcfmlp_output = self.ngcfmlp(userIdx, itemIdx) # (, embsize)
        mlp_output = self.mlp(userIdx,itemIdx) # (, embedSize)

        concat = torch.cat((ngcfmlp_output, mlp_output), -1)
        
        return self.output(concat).flatten()

class NGCFMLP_concat_MF_MLP(Module):
# Neural Graph collaborative filtering with mlp output concat with mlp
    def __init__(self,userNum,itemNum,adj,embedSize,layers):

        super(NGCFMLP_concat_MF_MLP, self).__init__()
    
        self.ngcfmlp = NGCFMLP_layer(userNum,itemNum,adj,embedSize,layers)
        self.mf = MF_layer(userNum, itemNum, embedSize)
        self.mlp = MLP_layer(userNum, itemNum, embedSize)

        self.output = nn.Linear(embedSize*3, 1)
        
        self._init_weight_()

    def _init_weight_(self):

        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self,userIdx,itemIdx):
        mf_output = self.mf(userIdx, itemIdx) # (, embsize)
        ngcfmlp_output = self.ngcfmlp(userIdx, itemIdx) # (, embsize)
        mlp_output = self.mlp(userIdx, itemIdx) # (, embedSize)

        concat = torch.cat((mf_output, ngcfmlp_output, mlp_output), -1)
        
        return self.output(concat).flatten()