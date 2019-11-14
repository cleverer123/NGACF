import time
import torch
from torch import nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.loaddata import load100KRatings, load1MRatings
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag
from tensorboardX import SummaryWriter

from graphattention.GACFmodel import GACF
from graphattention.GACFmodel import SVD
from graphattention.GACFmodel import NCF
from graphattention.dataPreprosessing import ML1K


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
    'epoch':30,
    'lr':0.01,
    'weight_decay': 0.001,
    'batch_size':2048,
    'train':0.7,
    'valid':0.15
}

ds = ML1K(rt)
trainLen = int(para['train']*len(ds))
validLen = int(para['valid']*len(ds))
train, valid, test = random_split(ds,[trainLen, validLen, len(ds)- validLen -trainLen])

train_loader = DataLoader(train, batch_size=para['batch_size'], shuffle=True,pin_memory=True)
valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=False,pin_memory=True)

model = GACF(userNum, itemNum, rt, 128, layers=[128,128,128]).cuda()
# model = SVD(userNum,itemNum,50).cuda()
# model = NCF(userNum,itemNum,64,layers=[128,64,32,16,8]).cuda()
optim = Adam(model.parameters(), lr=para['lr'],weight_decay=para['weight_decay'])
lossfn = MSELoss()

def train(model, train_loader, optim, lossfn):
    model.train()
    total_loss = 0.0
    for _, batch in enumerate(train_loader):
        optim.zero_grad()
        prediction = model(batch[0].cuda(), batch[1].cuda())
        loss = lossfn(batch[2].float().cuda(),prediction)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def valid(model, valid_loader, lossfn):
    model.eval()
    total_loss = 0.0
    for _, batch in enumerate(valid_loader):
        prediction = model(batch[0].cuda(), batch[1].cuda())
        loss = lossfn(batch[2].float().cuda(),prediction)
        total_loss += loss.item()
    return total_loss/len(valid_loader)


# Add summaryWriter. Results are in ./runs/. Run 'tensorboard --logdir=./runs' and see in browser.
summaryWriter = SummaryWriter(comment='lr:{}wd:{}'.format(para['lr'], para['weight_decay']))

for epoch in range(para['epoch']):
    epoch_loss = 0.0
    start_time = time.time()
    train_loss = train(model, train_loader, optim, lossfn)
    valid_loss = valid(model, valid_loader, lossfn)
    summaryWriter.add_scalar('loss/train_loss', train_loss, epoch)
    summaryWriter.add_scalar('loss/valid_loss', valid_loss, epoch)
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
    print('epoch:{}, train_loss:{}, valid_loss:{}'.format(epoch, train_loss, valid_loss))


test_loader = DataLoader(test,batch_size=len(test),)
test_loss = valid(model, test_loader, lossfn)
print('test_loss:', test_loss)


# for data in testdl:
#     prediction = model(data[0].cuda(),data[1].cuda())

# loss = lossfn(data[2].float().cuda(),prediction)
# print('testloss:', loss.item()) # MSEloss

