import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from data.mldataset import ItemDataSet
import multiprocessing
import heapq
import graphattention.metrics as metrics
############################################ Train ####################################################
def train(model, train_loader, optim, lossfn):
    
    model.train()
    
    total_loss = 0.0
    for batch_id, (user_idxs, item_idxs, labels) in enumerate(train_loader):
        # print(user_idxs, pos_item_idxs, neg_item_idxs)
        user_idxs = user_idxs.long().cuda()
        item_idxs = item_idxs.long().cuda()
        labels =  labels.float().cuda()

        optim.zero_grad()
        predictions = model(user_idxs, item_idxs) 
        
        loss = lossfn(predictions, labels)
        loss.sum().backward()
        optim.step()
        total_loss += loss.sum().item()
        if batch_id % 60 == 0 :
            print("The timeStamp of training batch {:03d}/{}".format(batch_id, len(train_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
            
    return total_loss/len(train_loader)

def train_bpr(model, train_loader, optim, lossfn):
    
    model.train()
    
    total_loss = 0.0
    for batch_id, (user_idxs, pos_item_idxs, neg_item_idxs) in enumerate(train_loader):
        # print(user_idxs, pos_item_idxs, neg_item_idxs)
        user_idxs = torch.LongTensor(user_idxs).cuda()
        pos_item_idxs = torch.LongTensor(pos_item_idxs).cuda()
        neg_item_idxs = torch.LongTensor(neg_item_idxs).cuda()

        optim.zero_grad()
        pos_scores = model(user_idxs, pos_item_idxs) 
        neg_scores = model(user_idxs, neg_item_idxs)
        loss = lossfn(pos_scores, neg_scores)
        loss.sum().backward()
        optim.step()
        total_loss += loss.sum().item()
        if batch_id % 60 == 0 :
            print("The timeStamp of training batch {:03d}/{}".format(batch_id, len(train_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
            
    return total_loss/len(train_loader)

############################################## Test ####################################################
def test(model, test_loader, lossfn):
    model.eval()
    total_loss = 0.0
    for _, batch in enumerate(test_loader):
        prediction = model(batch[0].long().cuda(), batch[1].long().cuda())
        loss = lossfn(prediction, batch[2].float().cuda())
        total_loss += loss.sum().item()
    return total_loss/len(test_loader)

########################################## Eval by Rank #################################################
from graphattention.evaluation import hit, ndcg
def eval_rank(model, test_loader, lossfn, parallel, top_k):
    model.eval()
    HR, NDCG = [], []
    for batch_id, batch in enumerate(test_loader):
        u_idxs = batch[0].long().cuda()
        i_idxs = batch[1].long().cuda()
        predictions = model(u_idxs, i_idxs)

        if parallel and torch.cuda.device_count() >1:
            i_idxs = i_idxs.view(torch.cuda.device_count(), -1)
            print(predictions)
            for device_idx, prediction in enumerate(predictions):
                device = torch.device('cuda:{}'.format(device_idx))
                i_idx = i_idxs[device_idx, :].to(device)
                _, indices = torch.topk(prediction, top_k)

                recommends = torch.take(i_idx, indices).cpu().numpy().tolist()
                gt_item = i_idx[0].item()
                HR.append(hit(gt_item, recommends))
                NDCG.append(ndcg(gt_item, recommends))
        else:
            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(i_idxs, indices).cpu().numpy().tolist()
            gt_item = i_idxs[0].item()
            HR.append(hit(gt_item, recommends))
            NDCG.append(ndcg(gt_item, recommends))
        
        
        if batch_id % 240 == 0 :
            print("The timeStamp of evaluating batch {:03d}/{}".format(batch_id, len(test_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
           

    return np.mean(HR), np.mean(NDCG)

########################################## Eval for bpr_loss #################################################
def eval_bpr(model, test_loader, test_user_num, itemNum, isparalell=False):
    model.eval()
    Ks = [10,20]
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    cores = multiprocessing.cpu_count() // 2
    # print('multiprocessing.cpu_count()', cores)
    pool = multiprocessing.Pool(cores)

    item_batch_size = 5000

    if isparalell:
        device_count = torch.cuda.device_count()
        item_loader = DataLoader(ItemDataSet(np.arange(itemNum)), batch_size=item_batch_size * device_count, shuffle=False, pin_memory=True)
    else:
        item_loader = DataLoader(ItemDataSet(np.arange(itemNum)), batch_size=item_batch_size, shuffle=False, pin_memory=True)
    
    for _, (user_batch, positive_items, negative_items) in enumerate(test_loader):

        u_idxs = user_batch.long().cuda()
        batch_ratings = []
        for b_idx, item_batch in enumerate(item_loader):
            i_idxs = item_batch.long().cuda()
            item_batch_ratings = model(u_idxs, i_idxs)
            # if b_idx ==0:
            #     print('item_batch_ratings', item_batch_ratings)

            if isparalell:
                batch_ratings.extend(item_batch_ratings.detach().cpu().numpy())
            else:
                batch_ratings.extend(item_batch_ratings.detach().cpu().numpy())

        batch_ratings = np.array(batch_ratings) # (batch_size, Item_num)
        # print('batch_ratings', batch_ratings.shape)
        user_batch_ratings = zip(batch_ratings, positive_items, negative_items)
        batch_metrics = pool.map(report_one_user, user_batch_ratings)

        for re in batch_metrics:
            result['precision'] += re['precision']/test_user_num
            result['recall'] += re['recall']/test_user_num
            result['ndcg'] += re['ndcg']/test_user_num
            result['hit_ratio'] += re['hit_ratio']/test_user_num
            result['auc'] += re['auc']/test_user_num

    pool.close()
    return result   

def report_one_user(x):
    Ks = [10, 20]
    # user u's ratings for user u, 
    ratings, positive_items, negative_items = x
    r, auc = ranklist_by_heapq(positive_items, negative_items, ratings, Ks)
    # if args.test_flag == 'part':
    #     r, auc = ranklist_by_heapq(list(positive_items), list(negative_items), ratings, Ks)
    # else:
    #     r, auc = ranklist_by_sorted(list(positive_items), list(negative_items), ratings, Ks)

    return get_performance(positive_items, r, auc, Ks)


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}