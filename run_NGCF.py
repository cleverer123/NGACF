import os
from os import path
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
from sklearn.model_selection import train_test_split

from graphattention.GACFmodel1 import GACFV1
from graphattention.GACFmodel2 import GACFV2
from graphattention.GACFMask import GACFMask
from graphattention.SPGA import SPGACF, MultiLayerSPGA, SPGAMGP
from graphattention.SPUIGACF import SPUIGACF, SPUIMultiGACF, SPUIGAGPCF
from graphattention.GACFmodel3 import GACFV3
from graphattention.GACFmodel4 import GACFV4
from graphattention.GACFmodel5 import GACFV5
from graphattention.GACFmodel6 import GACFV6

from graphattention.GCFmodel import  NGCFMF, NGCFMLP, NGCFMFMLP, NGCFMF_concat_MF, NGCFMF_concat_MLP, NGCFMF_concat_MF_MLP, NGCFMLP_concat_MF, NGCFMLP_concat_MLP, NGCFMLP_concat_MF_MLP
from graphattention.GCFModified import NGCFMF_M
from graphattention.GCFmodel import NCF
from graphattention.NMF import NMF

from graphattention.BPRLoss import BPRLoss

from train_eval_NGCF import eval_neg_sample, eval_neg_all, train_bpr, train_neg_sample
from data.loadGowalla import load1MRatings, load100KRatings, split_loo, loadGowalla, train_pos_neg_exclude_test, test_positives, test_positives_negtives, positives_negtives, get_adj_mat

from parallel import DataParallelModel, DataParallelCriterion, DataParallelCriterion2
CUDA_LAUNCH_BLOCKING=1

def prepareData(args):
    # train_data, test_data, userNum, itemNum, adj, test_user_num = load_data(args.dataset, args.evaluate, args.train, args.adj_type)
    # test_batch_size = len(test_data) if 'MSE' == args.evaluate else 100

    if args.dataset == 'Gowalla':
        datapath = path.dirname(__file__) + './data/' + args.dataset
        rt, train_df, test_df = loadGowalla(datapath)
        userNum = rt['userId'].max() + 1  
        itemNum = rt['itemId'].max() + 1
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
    elif args.dataset == 'ml1m':
        datapath = path.dirname(__file__) + './data/1M'
        rt = load1MRatings(datapath)
        userNum = rt['userId'].max()
        itemNum = rt['itemId'].max()
        rt['userId'] = rt['userId'] - 1
        rt['itemId'] = rt['itemId'] - 1
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
        if args.train_mode == 'NegSampling':
            train_df, test_df = split_loo(rt)
        elif args.train_mode == 'PairSampling':
            train_df, test_df = train_test_split(rt, test_size=0.2)
    elif args.dataset == 'ml100k':
        datapath = path.dirname(__file__) + './data/1K'
        rt = load100KRatings(datapath)
        # ml100k index starts at 1
        userNum = rt['userId'].max()
        itemNum = rt['itemId'].max()
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
        rt['userId'] = rt['userId'] - 1
        rt['itemId'] = rt['itemId'] - 1
        if args.train_mode == 'NegSampling':
            train_df, test_df = split_loo(rt)
        elif args.train_mode == 'PairSampling':
            train_df, test_df = train_test_split(rt, test_size=0.2)

    adj = get_adj_mat(datapath, rt, userNum, itemNum, args.adj_type)

    if args.train_mode == 'PairSampling' and args.eval_mode == 'AllNeg':
        train_pos_neg = train_pos_neg_exclude_test(rt, train_df)
        # test_pos_neg, _ = test_positives_negtives(test_df, train_pos_neg)
        test_df = test_positives(test_df) 
        test_pos_neg = train_pos_neg
        
    elif args.train_mode == 'NegSampling' and args.eval_mode == 'SampledNeg':
        train_pos_neg = positives_negtives(rt)
        test_pos_neg = train_pos_neg
    print('lenth of traindf', len(train_df), 'lenth of test_df', len(test_df))

    return train_df, test_df, train_pos_neg, test_pos_neg, userNum, itemNum, adj


def createModels(args, userNum, itemNum, adj):
    if args.model == 'NGCFMF':
        model = NGCFMF(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers).cuda()
    elif args.model == 'NGCFMF_M':
        model = NGCFMF_M(userNum, itemNum, adj, embedSize=args.embedSize, layers=args.layers).cuda()

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
def main(args):
    # Add summaryWriter. Results are in ./runs/ Run 'tensorboard --logdir=.' and see in browser.
    summaryWriter = SummaryWriter(comment='_DS:{}_M:{}_E:{}_L:{}_lr:{}_wd:{}_dp:{}_rs:{}_parallel:{}'.
                    format(args.dataset, args.model, args.embedSize, args.layers, args.lr, 
                    args.weight_decay, args.droprate, args.seed, args.parallel))
    train_df, test_df, train_pos_neg, test_pos_neg, userNum, itemNum, adj = prepareData(args)
    model, lossfn, optim = createModels(args, userNum, itemNum, adj)
    
    if args.resume_from:
        checkpoint = torch.load('ckpts/{}_{}_{:03d}.pkl'.format(args.model, args.dataset, args.resume_from))
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        print("=> loaded checkpoint '{}'".format('ckpts/{}_{:03d}.pkl'.format(args.model, args.resume_from)))

    for epoch in range(args.resume_from, args.epochs):
        t0 = time.time()
        if args.train_mode == 'PairSampling':
            train_loss = train_bpr(model, args.batch_size, train_df, train_pos_neg, optim, lossfn, args.parallel)
        elif args.train_mode == 'NegSampling':
            train_loss = train_neg_sample(model, args.batch_size, train_df, train_pos_neg, optim, lossfn, args.parallel)
        summaryWriter.add_scalar('loss/train_loss', train_loss, epoch)
        print('------epoch:{}, train_loss:{:5f}, time consuming:{}s'.format(epoch, train_loss, time.strftime("%H: %M: %S", time.gmtime(time.time() - t0))))
        if (epoch+1) % args.save_every == 0 :
            torch.save({'model': model.state_dict(),'optim': optim.state_dict()}, 'ckpts/{}_{}_{:03d}.pkl'.format(args.model, args.dataset, epoch+1))
        if (epoch+1) % args.eval_every == 0 :      

            t0 = time.time()
            if args.eval_mode == 'AllNeg':
                metrics = eval_neg_all(model, args.batch_size, test_df, test_pos_neg, itemNum, args.parallel)
                print('epoch:{} metrics:{}'.format(epoch, metrics))
                for i, K in enumerate([10,20]):
                    summaryWriter.add_scalar('metrics@{}/precision'.format(K), metrics['precision'][i], epoch)
                    summaryWriter.add_scalar('metrics@{}/recall'.format(K), metrics['recall'][i], epoch)
                    summaryWriter.add_scalar('metrics@{}/ndcg'.format(K), metrics['ndcg'][i], epoch)
                    summaryWriter.add_scalar('metrics@{}/hit_ratio'.format(K), metrics['hit_ratio'][i], epoch)
                    # summaryWriter.add_scalar('metrics@{}/auc'.format(K), metrics['auc'], epoch)
            elif args.eval_mode == 'SampledNeg':
                HR, NDCG = eval_neg_sample(model, args.batch_size, test_df, test_pos_neg, adj, 10, args.parallel)
                print('epoch:{}, HR:{:5f}, NDCG:{:5f}'.format(epoch, HR, NDCG))
                summaryWriter.add_scalar('metrics/HR', HR, epoch)
                summaryWriter.add_scalar('metrics/NDCG', NDCG, epoch)
            print("The time of evaluate epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - t0)))
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Graph Attention Collaborative Filtering')
    parser.add_argument("--dataset", type=str, default="ml100k", help="which dataset to use[ml100k/ml1m/Amazon])")  
    parser.add_argument("--model", type=str, default="SPUIGACF", help="which model to use(SPUIGACF, SPUIMultiGACF, SPUIGAGPCF)")
    parser.add_argument("--adj_type", type=str, default="ui_mat", help="which adj matrix to use [ui_mat, plain_adj, norm_adj, mean_adj]")
    parser.add_argument("--epochs", type=int, default=200, help="training epoches")
    parser.add_argument("--eval_every", type=int, default=5, help="evaluate every")
    parser.add_argument("--resume_from", type=int, default=0, help="resume from epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="weight_decay")
    parser.add_argument("--batch_size", type=int, default=2048, help="input batch size for training")
    parser.add_argument("--droprate", type=float, default=0.1, help="the rate for dropout")
    parser.add_argument("--train_rate", type=float, default=0.7, help="the train rate of dataset")
    parser.add_argument("--seed", type=int, default=2019, help="the seed for random")
    parser.add_argument("--embedSize", type=int, default=64, help="the size for Embedding layer")
    parser.add_argument("--layers", type=ast.literal_eval, default=[64,64], help="the layer list for propagation")
    parser.add_argument("--train_mode", type=str, default="NegSampling", help="the mode to train model [PairSampling, NegSampling]")
    parser.add_argument("--eval_mode", type=str, default="SampledNeg", help="the mode for evaluate[AllNeg, SampledNeg]")
    parser.add_argument("--parallel", type=ast.literal_eval, default=False, help="whether to use parallel model, input should be either 'True' or 'False'.")
    parser.add_argument("--gpu_id", type=str, default="0", help="default gpu index")
    args = parser.parse_args()
    if args.parallel:
        print('----------------Parallel Mode is enabled----------------')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 
    else:
        print('----------------Parallel Mode is disabled.----------------')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    main(args)

