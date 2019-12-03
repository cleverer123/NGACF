import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from data.mldataset import ItemDataSet
import multiprocessing
import heapq
import graphattention.metrics as metrics
from graphattention.BPRLoss import BPRLoss
from parallel import DataParallelCriterion2
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

#####################
def train_neg_sample(model, train_loader, optim, lossfn):
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
            print("The timeStamp of training batch {:03d}/{}".format(batch_id, len(train_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
            
    return total_loss/len(train_loader)

def train_bpr(model, train_loader, optim, lossfn):
    # lossfn = BPRLoss()
    model.train()
    # if isparalell and torch.cuda.device_count()>1:  
    #     lossfn = DataParallelCriterion2(lossfn)
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
        loss.backward()
        # # Ref: https://discuss.pytorch.org/t/is-the-loss-function-paralleled-when-using-dataparallel/3346/5
        # loss.backward(torch.ones(torch.cuda.device_count()).cuda())
        optim.step()
        # total_loss += loss.sum().item()
        total_loss += loss.item()
        # if batch_id % 60 == 0 :
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

########################################## Eval for test data with negtive samples #################################################
def eval_neg_sample(model, test_loader, test_user_num, top_k, is_parallel):
    model.eval()
    HR, NDCG = [], []
    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)
    for batch_id, (userIdx, pos_itemIdx, neg_itemIdxs) in enumerate(test_loader):
        # test_loader data structue :
        #    userIdx             pos_itemIdxs        neg_itemIdxs(neg_sample_size=3)
        #  tensor([1,         tensor([i1,          [tensor([i2,i4,i5]), 
        #          1,                 i3,           tensor([i2,i4,i5]),
        #          3])                i2])          tensor([i1,i3,i5])]
        # userIdx = userIdx.long().cuda()     
        pos_itemIdx = pos_itemIdx.reshape(-1,1)  # (batch_size, 1)         
        neg_itemIdxs = torch.stack(neg_itemIdxs).transpose(1,0)  # (batch_size, 99)    
        itemIdxs = torch.cat([pos_itemIdx, neg_itemIdxs], dim=1) # (batch_size, 100)

        u_Idxs = userIdx.expand(itemIdxs.shape[1], userIdx.shape[0]).transpose(1, 0).reshape(-1).long().cuda() #(batch_size * 100)
        i_Idxs = itemIdxs.reshape(-1).long().cuda()
        predictions = model(u_Idxs, i_Idxs)
        
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
        
        HR.extend(res[:].tolist())
        NDCG.extend(res[:,1].tolist())
        
        if batch_id % 240 == 0 :
            print("The timeStamp of evaluating batch {:03d}/{}".format(batch_id, len(test_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
    pool.close()
    return np.mean(HR), np.mean(NDCG)

def report_pos_neg(x):
    # print(x)
    # pos_itemIdx, recommends = x
    # recommends = list(recommends)
    # return hit(pos_itemIdx, recommends), ndcg(pos_itemIdx, recommends) 
    pos_itemIdx, itemIdxs, indices = x
    pos_itemIdx = int(pos_itemIdx)
    recommends = torch.take(itemIdxs, indices)
    recommends = list(recommends)
    return hit(pos_itemIdx, recommends), ndcg(pos_itemIdx, recommends) 


########################################## Eval for test data with all negtive #################################################
def eval_neg_all(model, test_loader, test_user_num, itemNum):
    model.eval()
    Ks = [10,20]
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    cores = multiprocessing.cpu_count() // 2
    # print('multiprocessing.cpu_count()', cores)
    pool = multiprocessing.Pool(cores)

    item_batch_size = 1000
    item_loader = DataLoader(ItemDataSet(np.arange(itemNum)), batch_size=item_batch_size, shuffle=False, pin_memory=False)
    
    for batch_id, (userIdx, pos_itemIdxs, neg_itemIdxs) in enumerate(test_loader):
        userIdx = userIdx.long().cuda()
        batch_ratings = []
        for _, itemIdx in enumerate(item_loader):
            itemIdx = itemIdx.long().cuda()
            u_Idxs = userIdx.expand(itemIdx.shape[0], userIdx.shape[0]).transpose(1, 0).reshape(-1)
            i_Idxs = itemIdx.expand(userIdx.shape[0], itemIdx.shape[0]).reshape(-1)

            item_batch_ratings = model(u_Idxs, i_Idxs)
            item_batch_ratings = item_batch_ratings.reshape(userIdx.shape[0], itemIdx.shape[0])

            batch_ratings.append(item_batch_ratings.detach().cpu().numpy())
        batch_ratings = np.concatenate(batch_ratings, axis=1) # (user_batch_size, Item_num)

        user_batch_ratings = zip(batch_ratings, pos_itemIdxs, neg_itemIdxs)
        batch_metrics = pool.map(report_one_user, user_batch_ratings)
        
        print("The timeStamp of test batch {:03d}/{}".format(batch_id, len(test_loader)) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time())))
        
        for re in batch_metrics:
            result['precision'] += re['precision']/test_user_num
            result['recall'] += re['recall']/test_user_num
            result['ndcg'] += re['ndcg']/test_user_num
            result['hit_ratio'] += re['hit_ratio']/test_user_num
            result['auc'] += re['auc']/test_user_num

    pool.close()
    return result   


# def eval_bpr_sigmoid(model, test_loader, test_user_num, itemNum, isparalell):
#     model.eval()
#     Ks = [10,20]
#     result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
#               'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

#     cores = multiprocessing.cpu_count() // 2
#     # print('multiprocessing.cpu_count()', cores)
#     pool = multiprocessing.Pool(cores)

#     item_batch_size = 5000

#     if isparalell:
#         device_count = torch.cuda.device_count()
#         item_loader = DataLoader(ItemDataSet(np.arange(itemNum)), batch_size=item_batch_size * device_count, shuffle=False, pin_memory=False)
#     else:
#         item_loader = DataLoader(ItemDataSet(np.arange(itemNum)), batch_size=item_batch_size, shuffle=False, pin_memory=False)
#     several_ratings_to_eval = []
#     for test_batch_idx, (user_batch, positive_items, negative_items) in enumerate(test_loader):
#         # print(user_batch, positive_items, negative_items)
#         # store ratings for paticular user: u. 
#         u_ratings = []
#         if isparalell and torch.cuda.device_count()>1:
#             pass
#         else:
#             u = user_batch.long().cuda()
#             for _, item_batch in enumerate(item_loader):
#                 u_idxs = torch.full_like(item_batch, u.item())
#                 i_idxs = item_batch.long().cuda()
#                 item_batch_ratings = model(u_idxs, i_idxs)
#                 u_ratings.append(item_batch_ratings.detach().cpu().numpy())
#             u_ratings = np.concatenate(u_ratings, axis=0) # (1, Item_num)
#         several_ratings_to_eval.append((u_ratings, positive_items, negative_items))
        
#         if (test_batch_idx + 1) % 5000 == 0:
#             batch_metrics = pool.map(report_one_user, several_ratings_to_eval)
#             for re in batch_metrics:
#                 result['precision'] += re['precision']/test_user_num
#                 result['recall'] += re['recall']/test_user_num
#                 result['ndcg'] += re['ndcg']/test_user_num
#                 result['hit_ratio'] += re['hit_ratio']/test_user_num
#                 result['auc'] += re['auc']/test_user_num
#                 several_ratings_to_eval = []
#     # deal with last batch
#     if len(several_ratings_to_eval) :
#         batch_metrics = pool.map(report_one_user, several_ratings_to_eval)
#         for re in batch_metrics:
#             result['precision'] += re['precision']/test_user_num
#             result['recall'] += re['recall']/test_user_num
#             result['ndcg'] += re['ndcg']/test_user_num
#             result['hit_ratio'] += re['hit_ratio']/test_user_num
#             result['auc'] += re['auc']/test_user_num

#     pool.close()
#     return result   


def report_one_user(x):
    Ks = [10, 20]
    # user u's ratings for user u, 
    ratings, positive_items, negative_items = x
    r, auc = ranklist_by_heapq(list(positive_items), list(negative_items), list(ratings), Ks)
    # if args.test_flag == 'part':
    #     r, auc = ranklist_by_heapq(list(positive_items), list(negative_items), ratings, Ks)
    # else:
    #     r, auc = ranklist_by_sorted(list(positive_items), list(negative_items), ratings, Ks)

    return get_performance(list(positive_items), r, auc, Ks)


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