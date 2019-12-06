import os
import ast
import time
import argparse 
import torch
from torch import nn as nn
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, DataParallel
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.loadPaircopy import load_data, load_data_adj
import pandas as pd
import numpy as np
from numpy import diag
from tensorboardX import SummaryWriter

from graphattention.GACFmodel1 import GACFV1
from graphattention.GACFmodel2 import GACFV2
from graphattention.GACFmodel3 import GACFV3
from graphattention.GACFmodel4 import GACFV4
from graphattention.GACFmodel5 import GACFV5, GACFV5_BPR
from graphattention.GACFmodel6 import GACFV6

from graphattention.GCFmodel import  GCF, GCF_MF
# from graphattention.GCFmodel import SVD
from graphattention.GCFmodel import NCF
from graphattention.BPRLoss import BPRLoss

from train_eval import eval_neg_sample, eval_neg_all

from parallel import DataParallelModel, DataParallelCriterion, DataParallelCriterion2

CUDA_LAUNCH_BLOCKING=1

def prepareData(args):
    # train_data, test_data, userNum, itemNum, adj, test_user_num = load_data(args.dataset, args.evaluate, args.train, args.adj_type)
    # test_batch_size = len(test_data) if 'MSE' == args.evaluate else 100

    train_data, test_data, userNum, itemNum, adj, test_user_num = load_data_adj(args.dataset, args.train_rate, args.adj_type, args.train_mode, args.eval_mode)

    if args.parallel == True :
        device_count = torch.cuda.device_count()
        train_loader = DataLoader(train_data, batch_size=args.batch_size * device_count, shuffle=True,pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    # 对于test_loader，SampledNeg模式下由于多线程处理受限于系统cpu性能，test_batch_size不宜过大，AllNeg模式下则是手动加载数据
    if args.eval_mode == 'AllNeg':
        test_loader = test_data
    else:
        test_loader = DataLoader(test_data, batch_size=args.batch_size // 8, shuffle=False, pin_memory=False)

    return train_loader, test_loader, userNum, itemNum, adj, test_user_num


def createModels(args, userNum, itemNum, adj):
    if args.model == 'NCF':
        model = NCF(userNum, itemNum, 64, layers=[128,64,32,16,8]).cuda()
    elif args.model == 'GCF':
        model = GCF(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers).cuda()
    elif args.model == 'GACFV1':
        model = GACFV1(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers, droprate=args.droprate).cuda()
    elif args.model == 'GACFV2':
        model = GACFV2(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers, droprate=args.droprate).cuda()
    elif args.model == 'GACFV3':
        model = GACFV2(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers, droprate=args.droprate).cuda()
    elif args.model == 'GACFV4':
        model = GACFV4(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers, droprate=args.droprate).cuda()
    elif args.model == 'GACFV5':
        model = GACFV5(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers, droprate=args.droprate).cuda()
    elif args.model == 'GACFV6':
        model = GACFV6(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers, droprate=args.droprate).cuda()

    if args.train_mode == 'PairSampling':
        lossfn = BPRLoss()
        if args.parallel == True :
            model = DataParallelModel(model)    
            lossfn = DataParallelCriterion2(lossfn)
    elif args.train_mode == 'NegSampling':
        lossfn = BCEWithLogitsLoss()
        if args.parallel == True :
            model = DataParallelModel(model)       # 并行化model
            lossfn = DataParallelCriterion(lossfn)       # 并行化损失函数
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, lossfn, optim


######################################## MAIN-TRAINING-TESTING #################################
# Add summaryWriter. Results are in ./runs/ Run 'tensorboard --logdir=.' and see in browser.
# def main(args):
#     summaryWriter = SummaryWriter(comment='_DS:{}_M:{}_Layer:{}_lr:{}_wd:{}_dp:{}_rs:{}'.
#                     format(args.dataset, args.model, len(args.layers), args.lr, 
#                     args.weight_decay, args.droprate, args.seed))

#     if args.evaluate in ['BPR', 'BPR_NegSample']:
#         train_data, test_data, userNum, itemNum, adj, test_user_num = load_data(args.dataset, args.evaluate, args.train, args.adj_type)
        
#         train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,pin_memory=True)
#         test_loader = DataLoader(test_data, batch_size=args.batch_size , shuffle=False, pin_memory=False)
        
#         model, lossfn, optim = createModels(args, userNum, itemNum, adj)
#         for epoch in range(args.epochs):
#             t0 = time.time()
#             train_loss = train_bpr(model, train_loader, optim, lossfn)
#             summaryWriter.add_scalar('loss/train_loss', train_loss, epoch)
#             print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - t0)))
#             print('------epoch:{}, train_loss:{:5f}'.format(epoch, train_loss))
#             if (epoch+1) % args.eval_every ==0 :
#                 if args.evaluate == 'BPR':
#                     metrics = eval_bpr(model, test_loader, test_user_num, itemNum)
#                     print('epoch:{} metrics:{}'.format(epoch, metrics))
#                     for i, K in enumerate([10,20]):
#                         summaryWriter.add_scalar('metrics@{}/precision'.format(K), metrics['precision'][i], epoch)
#                         summaryWriter.add_scalar('metrics@{}/recall'.format(K), metrics['recall'][i], epoch)
#                         summaryWriter.add_scalar('metrics@{}/ndcg'.format(K), metrics['ndcg'][i], epoch)
#                         summaryWriter.add_scalar('metrics@{}/auc'.format(K), metrics['auc'], epoch)
#                 elif args.evaluate == 'BPR_NegSample':
#                     metrics = eval_bpr_negsample(model, test_loader, 20)
#                     print('epoch:{} HR, NDCG:{}'.format(epoch, metrics))

#                 # metrics = eval_bpr_sigmoid(model, test_loader, test_user_num, itemNum, args.parallel) 
                
                
#     else:
#         train_loader, test_loader, userNum, itemNum, adj = prepareData(args)
#         model, lossfn, optim = createModels(args, userNum, itemNum, adj)

#         for epoch in range(args.epochs):
#             start_time = time.time()
            
#             train_loss = train(model, train_loader, optim, lossfn)
#             summaryWriter.add_scalar('loss/train_loss', train_loss, epoch)
#             print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
#             print('------epoch:{}, train_loss:{:5f}'.format(epoch, train_loss))
#             if (epoch+1) % args.eval_every ==0 :
#                 if args.evaluate == 'MSE':
#                     test_loss = test(model, test_loader, lossfn)
#                     summaryWriter.add_scalar('loss/test_loss', test_loss, epoch)
#                     print('------epoch:{}, test_loss:{:5f}'.format(epoch, test_loss))

#                 if args.evaluate == 'RANK':
#                     start_time = time.time()
#                     HR, NDCG = eval_rank(model, test_loader, lossfn, args.parallel, 10)
#                     summaryWriter.add_scalar('metrics/HR', HR, epoch)
#                     summaryWriter.add_scalar('metrics/NDCG', NDCG, epoch)
#                     print("The time of evaluate epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
#                     print('epoch:{}, HR:{:5f}, NDCG:{:5f}'.format(epoch, HR, NDCG))

######################################## MAIN-TRAINING-TESTING #################################
def main(args):
    # Add summaryWriter. Results are in ./runs/ Run 'tensorboard --logdir=.' and see in browser.
    summaryWriter = SummaryWriter(comment='_DS:{}_M:{}_Layer:{}_lr:{}_wd:{}_dp:{}_rs:{}'.
                    format(args.dataset, args.model, len(args.layers), args.lr, 
                    args.weight_decay, args.droprate, args.seed))
    train_loader, test_loader, userNum, itemNum, adj, test_user_num = prepareData(args)
    model, lossfn, optim = createModels(args, userNum, itemNum, adj)

    if args.train_mode == 'PairSampling':
        from train_eval import train_bpr as train_
    elif args.train_mode == 'NegSampling':
        # from train_eval import train as train_ train_neg_sample
        from train_eval import train_neg_sample as train_
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_(model, train_loader, optim, lossfn)
        summaryWriter.add_scalar('loss/train_loss', train_loss, epoch)
        print('------epoch:{}, train_loss:{:5f}, time consuming:{}s'.format(epoch, train_loss, time.strftime("%H: %M: %S", time.gmtime(time.time() - t0))))
        if (epoch+1) % args.eval_every == 0 :
            t0 = time.time()
            if args.eval_mode == 'AllNeg':
                metrics = eval_neg_all(model, test_loader, test_user_num, itemNum, args.parallel)
                print('epoch:{} metrics:{}'.format(epoch, metrics))
                for i, K in enumerate([10,20]):
                    summaryWriter.add_scalar('metrics@{}/precision'.format(K), metrics['precision'][i], epoch)
                    summaryWriter.add_scalar('metrics@{}/recall'.format(K), metrics['recall'][i], epoch)
                    summaryWriter.add_scalar('metrics@{}/ndcg'.format(K), metrics['ndcg'][i], epoch)
                    summaryWriter.add_scalar('metrics@{}/hit_ratio'.format(K), metrics['hit_ratio'], epoch)
                    summaryWriter.add_scalar('metrics@{}/auc'.format(K), metrics['auc'], epoch)
            elif args.eval_mode == 'SampledNeg':
                HR, NDCG = eval_neg_sample(model, test_loader, test_user_num, 10, args.parallel)
                print('epoch:{}, HR:{:5f}, NDCG:{:5f}'.format(epoch, HR, NDCG))
                summaryWriter.add_scalar('metrics/HR', HR, epoch)
                summaryWriter.add_scalar('metrics/NDCG', NDCG, epoch)
            print("The time of evaluate epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - t0)))
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Graph Attention Collaborative Filtering')
    parser.add_argument("--dataset", type=str, default="Gowalla", help="which dataset to use[ml100k/ml1m/Amazon])")  
    parser.add_argument("--model", type=str, default="GCF", help="which model to use(NCF/GCF/GACFV1/GACFV2/GACFV3/GACFV4/GACFV5/GACFV6)")
    parser.add_argument("--adj_type", type=str, default="mean_adj", help="which adj matrix to use [plain_adj, norm_adj, mean_adj]")
    parser.add_argument("--epochs", type=int, default=300, help="training epoches")
    parser.add_argument("--eval_every", type=int, default=50, help="evaluate every")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight_decay")
    parser.add_argument("--batch_size", type=int, default=2048, help="input batch size for training")
    parser.add_argument("--droprate", type=float, default=0.1, help="the rate for dropout")
    parser.add_argument("--train_rate", type=float, default=0.7, help="the train rate of dataset")
    parser.add_argument("--seed", type=int, default=2019, help="the seed for random")
    parser.add_argument("--embedSize", type=int, default=64, help="the size for Embedding layer")
    parser.add_argument("--layers", type=list, default=[64,64], help="the layer list for propagation")
    parser.add_argument("--train_mode", type=str, default="NegSampling", help="the mode to train model [PairSampling, NegSampling]")
    parser.add_argument("--eval_mode", type=str, default="SampledNeg", help="the mode for evaluate[AllNeg, SampledNeg]")
    parser.add_argument("--parallel", type=ast.literal_eval, default=False, help="whether to use parallel model, input should be either 'True' or 'False'.")
    args = parser.parse_args()
    if args.parallel:
        print('----------------Parallel Mode is enabled----------------')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2' 
    else:
        print('----------------Parallel Mode is disabled.----------------')
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)

