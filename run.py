import time
import torch
from torch import nn as nn
from data.loaddata import load100KRatings, load1MRatings
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag
from graphattention.GACFmodel import GACF
from torch.utils.data import DataLoader
from graphattention.dataPreprosessing import ML1K
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from graphattention.GACFmodel import SVD
from graphattention.GACFmodel import NCF

# import os
# os.chdir('/home/chenliu/DeepGraphFramework/NeuralCollaborativeFiltering/NGCF-pytorch')
# rt = load1MRatings()
rt = load100KRatings()
userNum = rt['userId'].max()
print('userNum', userNum)
itemNum = rt['itemId'].max()
print('itemNum', itemNum)

rt['userId'] = rt['userId'] - 1
rt['itemId'] = rt['itemId'] - 1
#
# rtIt = rt['itemId'] + userNum
# uiMat = coo_matrix((rt['rating'],(rt['userId'],rt['itemId'])))
# uiMat_upperPart = coo_matrix((rt['rating'],(rt['userId'],rtIt)))
# uiMat = uiMat.transpose()
# uiMat.resize((itemNum,userNum+itemNum))
# uiMat = uiMat.todense()
# uiMat_t = uiMat.transpose()
# zeros1 = np.zeros((userNum,userNum))
# zeros2 = np.zeros((itemNum,itemNum))
#
# p1 = np.concatenate([zeros1,uiMat],axis=1)
# p2 = np.concatenate([uiMat_t,zeros2],axis=1)
# mat = np.concatenate([p1,p2])
#
# count = (mat > 0)+0
# diagval = np.array(count.sum(axis=0))[0]
# diagval = np.power(diagval,(-1/2))
# D_ = diag(diagval)
#
# L = np.dot(np.dot(D_,mat),D_)
#
para = {
    'epoch':60,
    'lr':0.01,
    'batch_size':256,
    'train':0.8
}

ds = ML1K(rt)
trainLen = int(para['train']*len(ds))
train,test = random_split(ds,[trainLen,len(ds)-trainLen])
dl = DataLoader(train,batch_size=para['batch_size'],shuffle=True,pin_memory=True)

model = GACF(userNum, itemNum, rt, 128, layers=[128,64,]).cuda()
# model = SVD(userNum,itemNum,50).cuda()
# model = NCF(userNum,itemNum,64,layers=[128,64,32,16,8]).cuda()
optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
lossfn = MSELoss()

for i in range(para['epoch']):
    epoch_loss = 0.0
    start_time = time.time()
    for id,batch in enumerate(dl):
        # print(id, len(dl))
        optim.zero_grad()
        prediction = model(batch[0].cuda(), batch[1].cuda())
        loss = lossfn(batch[2].float().cuda(),prediction)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        # print(loss.item())
    print("The time elapse of epoch {:03d}".format(i) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
    print('epoch:{},loss:{}'.format(i, epoch_loss/id))


testdl = DataLoader(test,batch_size=len(test),)
for data in testdl:
    prediction = model(data[0].cuda(),data[1].cuda())

loss = lossfn(data[2].float().cuda(),prediction)
print('testloss:', loss.item()) # MSEloss

