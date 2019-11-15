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

from graphattention.dataPreprosessing import ML1K

from graphattention.GCFmodel import  GCF
# from graphattention.GCFmodel import SVD
# from graphattention.GCFmodel import NCF

# import os
# os.chdir('/home/chenliu/DeepGraphFramework/NeuralCollaborativeFiltering/NGCF-pytorch')
# rt = load1MRatings()


para = {
    'model': 'GACF', 
    'epoch': 50,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 2048,
    'droprate': 0.3,
    'train': 0.7,
    'valid': 0.15,
    'seed': 2019
}

torch.manual_seed(para['seed'])
np.random.seed(para['seed'])

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


ds = ML1K(rt)
trainLen = int(para['train']*len(ds))
validLen = int(para['valid']*len(ds))
train, valid, test = random_split(ds,[trainLen, validLen, len(ds)- validLen -trainLen])

train_loader = DataLoader(train, batch_size=para['batch_size'], shuffle=True,pin_memory=True)
valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=False,pin_memory=True)

if para['model'] == 'GACF':
    model = GACF(userNum, itemNum, rt, 128, layers=[128,128,128], droprate=para['droprate']).cuda()
elif para['model'] == 'GCF':
    model = GCF(userNum, itemNum, rt, 128, layers=[128,128,128]).cuda()
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
summaryWriter = SummaryWriter(comment='_M:{}_lr:{}_wd:{}_dp:{}_rs:{}'.format(para['model'], para['lr'], para['weight_decay'], para['droprate'], para['seed']))
best_valid = 1
best_model = None
for epoch in range(para['epoch']):
    epoch_loss = 0.0
    start_time = time.time()
    train_loss = train(model, train_loader, optim, lossfn)
    valid_loss = valid(model, valid_loader, lossfn)
    summaryWriter.add_scalar('loss/train_loss', train_loss, epoch)
    summaryWriter.add_scalar('loss/valid_loss', valid_loss, epoch)
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
    print('epoch:{}, train_loss:{}, valid_loss:{}'.format(epoch, train_loss, valid_loss))

    if best_valid > valid_loss:
        best_valid, best_epoch = valid_loss, epoch
        # RuntimeError: sparse tensors do not have storage
        # torch.save(model, 'best_models/M{}_lr{}_wd{}.model'.format(para['model'], para['lr'], para['weight_decay']))

# print('best_model at epoch:{} with valid_loss:{}'.format(best_epoch, best_valid))

test_loader = DataLoader(test,batch_size=len(test),)
test_loss = valid(model, test_loader, lossfn)
print('test_loss:', test_loss)
# summaryWriter.add_scalar('loss/test_loss', test_loss, epoch)


