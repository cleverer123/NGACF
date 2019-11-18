import time
import torch
from torch import nn as nn
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.loaddata import load100KRatings, load1MRatings, load_data
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag
from tensorboardX import SummaryWriter

from graphattention.evaluation import hit, ndcg

from graphattention.GACFmodel1 import GACFV1
from graphattention.GACFmodel2 import GACFV2
from graphattention.GACFmodel3 import GACFV3
from graphattention.GACFmodel4 import GACFV4
from graphattention.GACFmodel5 import GACFV5
from graphattention.GACFmodel6 import GACFV6

from graphattention.GCFmodel import  GCF

# from graphattention.GCFmodel import SVD
from graphattention.GCFmodel import NCF

# import os
# os.chdir('/home/chenliu/DeepGraphFramework/NeuralCollaborativeFiltering/NGCF-pytorch')


######################################## PARAMETERS #################################
para = {
    'dataset': 'ml100k', #[ml100k, ml1m]
    'model': 'GCF', #[NCF, GCF, GACFV1, GACFV2, GACFV3, GACFV4, GACFV5, GACFV6]
    'epoch': 50,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 2048,
    'droprate': 0.1,
    'train': 0.7,
    # 'valid': 0.15,
    'seed': 2019,
    'layers': [128,128,],
    'embedSize': 128,
    'evaluate': 'RANK' # [MSE, RANK]
}


torch.manual_seed(para['seed'])
np.random.seed(para['seed'])

######################################## PREPARE DATA #################################

rt, train_data, test_data, userNum, itemNum = load_data(para['dataset'], para['evaluate'], para['train'])


test_batch_size = len(test_data) if 'MSE' == para['evaluate'] else 100

train_loader = DataLoader(train_data, batch_size=para['batch_size'], shuffle=True,pin_memory=True)
test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True)


######################################## CREATE MODELS #################################
if para['model'] == 'NCF':
    model = NCF(userNum, itemNum, 64, layers=[128,64,32,16,8]).cuda()
elif para['model'] == 'GCF':
    model = GCF(userNum, itemNum, rt, embedSize=para['embedSize'], layers=para['layers']).cuda()
elif para['model'] == 'GACFV1':
    model = GACFV1(userNum, itemNum, rt, embedSize=para['embedSize'], layers=para['layers'], droprate=para['droprate']).cuda()
elif para['model'] == 'GACFV2':
    model = GACFV2(userNum, itemNum, rt, embedSize=para['embedSize'], layers=para['layers'], droprate=para['droprate']).cuda()
elif para['model'] == 'GACFV3':
    model = GACFV2(userNum, itemNum, rt, embedSize=para['embedSize'], layers=para['layers'], droprate=para['droprate']).cuda()
elif para['model'] == 'GACFV4':
    model = GACFV4(userNum, itemNum, rt, embedSize=para['embedSize'], layers=para['layers'], droprate=para['droprate']).cuda()
elif para['model'] == 'GACFV5':
    model = GACFV5(userNum, itemNum, rt, embedSize=para['embedSize'], layers=para['layers'], droprate=para['droprate']).cuda()
elif para['model'] == 'GACFV6':
    model = GACFV6(userNum, itemNum, rt, embedSize=para['embedSize'], layers=para['layers'], droprate=para['droprate']).cuda()
# model = SVD(userNum,itemNum,50).cuda()

# model = NCF(userNum,itemNum,64,layers=[128,64,32,16,8]).cuda()
optim = Adam(model.parameters(), lr=para['lr'], weight_decay=para['weight_decay'])
if para['evaluate'] == 'MSE':
    lossfn = MSELoss()
if para['evaluate'] == 'RANK':
    lossfn = BCEWithLogitsLoss()

######################################## TRAINING #################################
def train(model, train_loader, optim, lossfn):
    model.train()
    total_loss = 0.0
    for _, batch in enumerate(train_loader):
        optim.zero_grad()
        prediction = model(batch[0].long().cuda(), batch[1].long().cuda())
        loss = lossfn(prediction, batch[2].float().cuda())
        # print("label",batch[2].float().cuda())
        # print("prediction",prediction)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def test(model, test_loader, lossfn):
    model.eval()
    total_loss = 0.0
    for _, batch in enumerate(test_loader):
        prediction = model(batch[0].long().cuda(), batch[1].long().cuda())
        loss = lossfn(prediction, batch[2].float().cuda())
        total_loss += loss.item()
    return total_loss/len(test_loader)

def eval_rank(model, test_loader, lossfn, top_k):
    model.eval()
    HR, NDCG = [], []
    for _, batch in enumerate(test_loader):
        u_idx = batch[0].long().cuda()
        i_idx = batch[1].long().cuda()
        prediction = model(u_idx, i_idx)
        _, indices = torch.topk(prediction, top_k)
        recommends = torch.take(i_idx, indices).cpu().numpy().tolist()
        gt_item = i_idx[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

# Add summaryWriter. Results are in ./runs/. Run 'tensorboard --logdir=./runs' and see in browser.
summaryWriter = SummaryWriter(comment='_DS:{}_M:{}_Layer:{}_lr:{}_wd:{}_dp:{}_rs:{}'.
                format(para['dataset'], para['model'], len(para['layers']), para['lr'], 
                para['weight_decay'], para['droprate'], para['seed']))
best_test = 1
best_model = None
for epoch in range(para['epoch']):
    epoch_loss = 0.0
    start_time = time.time()
    train_loss = train(model, train_loader, optim, lossfn)
    test_loss = test(model, test_loader, lossfn)

    summaryWriter.add_scalar('loss/train_loss', train_loss, epoch)
    summaryWriter.add_scalar('loss/test_loss', test_loss, epoch)
    
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
    print('epoch:{}, train_loss:{:5f}, test_loss:{:5f}'.format(epoch, train_loss, test_loss))

    if para['evaluate'] == 'RANK':
        start_time = time.time()
        HR, NDCG = eval_rank(model, test_loader, lossfn, 10)
        summaryWriter.add_scalar('metrics/HR', HR, epoch)
        summaryWriter.add_scalar('metrics/NDCG', NDCG, epoch)
        print("The time of evaluate epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
        print('epoch:{}, HR:{:5f}, NDCG:{:5f}'.format(epoch, HR, NDCG))
    

    if best_test > test_loss:
        best_test, best_epoch = test_loss, epoch
        # RuntimeError: sparse tensors do not have storage
        # torch.save(model, 'best_models/M{}_lr{}_wd{}.model'.format(para['model'], para['lr'], para['weight_decay']))

print('best_model at epoch:{} with test_loss:{}'.format(best_epoch, best_test))


# print('test_loss:', test_loss)
# summaryWriter.add_scalar('loss/test_loss', test_loss, epoch)


