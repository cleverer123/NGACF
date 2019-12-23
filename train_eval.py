import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from data.mldataset import ItemDataSet
import multiprocessing
import heapq
import graphattention.metrics as metrics
from graphattention.BPRLoss import BPRLoss
from graphattention.evaluation import hit, ndcg

#################################### train_neg_sample ###################################
def train_neg_sample(model, train_loader, optim, lossfn, is_parallel):
    model.train()
    total_loss = 0.0

    for batch_id, (userIdx, pos_itemIdx, neg_itemIdxs) in enumerate(train_loader):
        # userIdx = userIdx.long().cuda() 
        pos_itemIdx = pos_itemIdx.reshape(-1,1) # (batch_size, 1)  
        neg_itemIdxs = torch.stack(neg_itemIdxs).transpose(1,0)  # (batch_size, 99) 
        itemIdxs = torch.cat([pos_itemIdx, neg_itemIdxs], dim=1) # (batch_size, 100)

        pos_labels = torch.ones_like(pos_itemIdx)
        neg_labels = torch.zeros_like(neg_itemIdxs)
        labels = torch.cat([pos_labels, neg_labels], dim=1)

        u_Idxs = userIdx.expand(itemIdxs.shape[1], userIdx.shape[0]).transpose(1, 0).reshape(-1).long().cuda()  #(batch_size * 100)
        i_Idxs = itemIdxs.reshape(-1).long().cuda() 
        labels = labels.reshape(-1).float().cuda()
        
        optim.zero_grad()
        predictions = model(u_Idxs, i_Idxs) 
        loss = lossfn(predictions, labels)
        loss.sum().backward()
        optim.step()
        total_loss += loss.sum().item()
        if batch_id % 60 == 0 :
            print("-----------The timeStamp of training batch {:03d}/{}".format(batch_id, len(train_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
            
    return total_loss/len(train_loader)

########################################## Eval for test data with negtive samples #################################################
def eval_neg_sample(model, test_loader, test_user_num, top_k, is_parallel):
    HR, NDCG = [], []
    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)

    with torch.no_grad():
        model.eval()
        for batch_id, (userIdx, pos_itemIdx, neg_itemIdxs) in enumerate(test_loader):
            # test_loader data structue :
            #    userIdx             pos_itemIdxs        neg_itemIdxs(neg_sample_size=3)
            #  tensor([1,         tensor([i1,         [tensor([i2,  tensor([i4,   tensor([i5,
            #          1,                 i3,                  i2,          i4,           i5,
            #          3])                i2])                 i1]),        i3]),         i5])]
            # userIdx = userIdx.long().cuda()     
            pos_itemIdx = pos_itemIdx.reshape(-1,1)  # (batch_size, 1)         
            neg_itemIdxs = torch.stack(neg_itemIdxs).transpose(1,0)  # (batch_size, 99)    
            itemIdxs = torch.cat([pos_itemIdx, neg_itemIdxs], dim=1) # (batch_size, 100)

            u_Idxs = userIdx.expand(itemIdxs.shape[1], userIdx.shape[0]).transpose(1, 0).reshape(-1).long().cuda() #(batch_size * 100)
            i_Idxs = itemIdxs.reshape(-1).long().cuda()
                
            predictions = model(u_Idxs, i_Idxs)        
            # 并行化处理
            if is_parallel and torch.cuda.device_count()>1:
                l = []
                for prediction in predictions:
                    l.append(prediction.detach().cpu())
                predictions = torch.cat(l, dim=0)
            else:
                predictions = predictions.detach().cpu()

            predictions = predictions.reshape(-1, itemIdxs.shape[1])
            _, indices = torch.topk(predictions, top_k, dim=1) # (batch_size, top_k)
            # x = zip(pos_itemIdx.cpu().numpy(), itemIdxs.cpu(), indices.cpu())
            x = zip(pos_itemIdx, itemIdxs, indices)
            res = pool.map(report_pos_neg, x)

            res = np.array(res)
            
            HR.extend(res[:,0].tolist())
            NDCG.extend(res[:,1].tolist())
            
            if batch_id % 240 == 0 :
                print("-----------The timeStamp of evaluating batch {:03d}/{}".format(batch_id, len(test_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
        pool.close()
    return np.mean(HR), np.mean(NDCG)

def report_pos_neg(x):
    pos_itemIdx, itemIdxs, indices = x
    pos_itemIdx = int(pos_itemIdx)
    recommends = torch.take(itemIdxs, indices)
    recommends = list(recommends)
    return hit(pos_itemIdx, recommends), ndcg(pos_itemIdx, recommends) 
