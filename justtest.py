# import torch
# def getSparseEye(num):
#     i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
#     val = torch.FloatTensor([1]*num)
#     return torch.sparse.FloatTensor(i,val)

# a = getSparseEye(6)
# print(a)



# from torch.nn import Module
# import torch
# class GNNLayer(Module):

#     def __init__(self,inF,outF):

#         super(GNNLayer,self).__init__()
#         self.inF = inF
#         self.outF = outF
#         self.linear = torch.nn.Linear(in_features=inF, out_features=outF)
#         self.interActTransform = torch.nn.Linear(in_features=inF, out_features=outF)

#     def forward(self, laplacianMat,selfLoop,features):
#         # for GCF ajdMat is a (N+M) by (N+M) mat
#         # laplacianMat L = D^-1(A)D^-1 # 拉普拉斯矩阵
#         L1 = laplacianMat + selfLoop
#         L2 = laplacianMat.cuda()
#         L1 = L1.cuda()
#         # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，
#         # 比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
#         inter_feature = torch.mul(features,features)  # 
#         # torch.mm(a, b)是矩阵a和b矩阵相乘，
#         # 比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
#         # torch.sparse.mm下面是稀疏矩阵相乘
#         inter_part1 = self.linear(torch.sparse.mm(L1,features))
#         inter_part2 = self.interActTransform(torch.sparse.mm(L2,inter_feature))

#         return inter_part1 + inter_part2


# a = GNNLayer(10,2)
# print(a)

import torch
import torch.nn as nn

uEmbd = nn.Embedding(8, 20)
print("nn.Embedding的结果",uEmbd)
uidx = torch.LongTensor([i for i in range(8)])
userEmbedding = uEmbd(uidx)
print(userEmbedding)

iEmbd = nn.Embedding(10, 20)
print("nn.Embedding的结果",iEmbd)
iidx = torch.LongTensor([i for i in range(10)])
itemEmbedding = iEmbd(iidx)
print(itemEmbedding)

features = torch.cat([userEmbedding,itemEmbedding],dim=0)
print("oooo",features[uidx])