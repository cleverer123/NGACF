import os
import time
from os import path
import random
import pandas as pd
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import random_split
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse.coo import coo_matrix 
from ast import literal_eval
from sklearn.model_selection import train_test_split

from data.mldataset import MLDataSet, PairDataset, TestDataSet
# from mldataset import MLDataSet, PairDataset, TestDataSet

# dataset:movielens100K
def load100KRatings(datapath):
    rt = pd.read_table(datapath + '/u.data',sep='\t',names=['userId','itemId','rating','timestamp'])
    return rt

# dataset:movielens1M
def load1MRatings(datapath):
    rt = pd.read_table(datapath + '/ratings.dat',sep='::',names=['userId','itemId','rating','timestamp'])
    return rt

# dataset:Amazonbook
def loadAmazonbook(datapath):
    train_df = pd.read_table(datapath + '/Amazon_train.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    test_df = pd.read_table(datapath + '/Amazon_test.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    # train_df.dropna(inplace=True)
    # test_df.dropna(inplace=True)
    rt = train_df.append(test_df)
    return rt, train_df, test_df

def sample_train_pair(train_df):
    # train_df: ['userId', 'positive_items', 'negative_items']
    sampled_batch = train_df.sample(n=len(train_df))
    # sample pairs
    sampled_batch['pos_sample'] = sampled_batch['positive_items'].apply(lambda x: random.sample(x, 1))
    sampled_batch['neg_sample'] = sampled_batch['negative_items'].apply(lambda x: random.sample(x, 1))

    return sampled_batch[['userId', 'pos_sample', 'neg_sample']]

# 构建正负样本集合：userId：int; positive_items：set; negative_items: set.
def df_positive_negtive(df):
    user_num = len(df['userId'].unique())
    item_pool = set(df['itemId'].unique())  # item_pool: 1682
    df = df.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'positive_items'})
    df['negative_items'] = df['positive_items'].apply(lambda x: item_pool - x)
    return  df[['userId', 'positive_items', 'negative_items']], user_num

def load_data(dataset, evaluate, ratio_train):
    if dataset == 'Amazon':
        datapath = path.dirname(__file__) + '/Amazon'
        rt, train_df, test_df = loadAmazonbook(datapath)
        
        userNum = rt['userId'].max()
        itemNum = rt['itemId'].max()
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
        
        rt['userId'] = rt['userId'] - 1
        rt['itemId'] = rt['itemId'] - 1
        train_df['userId'] = train_df['userId'] - 1
        train_df['itemId'] = train_df['itemId'] - 1

        if evaluate == 'BPR':
            train_df, train_user_num = df_positive_negtive(train_df)
            train_data = sample_train_pair(train_df)
            train_data = PairDataset(train_data.values)

            # TODO: Generate Test_data
            test_df, test_user_num = df_positive_negtive(test_df)
            test_data = TestDataSet(test_df)
            print('train_user_num:{}, test_user_num:{}'.format(train_user_num, test_user_num))
            
        else:
            print('Amazon dataset is too large to process with evaluate'.format(evaluate))

    else:
        if dataset == 'ml1m':
            datapath = path.dirname(__file__) + '/1M'
            rt = load1MRatings(datapath)
        elif dataset == 'ml100k':
            datapath = path.dirname(__file__) + '/1K'
            rt = load100KRatings(datapath)
        
        userNum = rt['userId'].max()
        itemNum = rt['itemId'].max()
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))

        rt['userId'] = rt['userId'] - 1
        rt['itemId'] = rt['itemId'] - 1
        
        train_user_num = userNum
        test_user_num = userNum

        if evaluate == 'MSE':
            ds = rt.values
            ds = MLDataSet(ds)
            trainLen = int(ratio_train*len(ds))
            train_data, test_data = random_split(ds, [trainLen, len(ds)-trainLen])
    
        elif evaluate == 'RANK':
            train_data, test_data = load_data_negsample(rt, dataset, ratio_train)
            train_data = MLDataSet(train_data)
            test_data = MLDataSet(test_data)

        elif evaluate == 'BPR':
            
            train_df, test_df = train_test_split(rt, test_size=1-ratio_train)
            # Generate train_data
            train_df, train_user_num = df_positive_negtive(train_df)
            train_data = sample_train_pair(train_df)
            train_data = PairDataset(train_data.values)
            # Generate test_data
            test_df, test_user_num = df_positive_negtive(test_df)
            test_data = TestDataSet(test_df.values)
            print('train_user_num:{}, test_user_num:{}'.format(train_user_num, test_user_num))
    
    adj_map = get_adj_mat(datapath, rt, userNum, itemNum)

    return train_data, test_data, userNum, itemNum, adj_map, train_user_num, test_user_num


def get_adj_mat(path, rt, userNum, itemNum):
    adj_map = {}
    try:
        t0 = time.time()
        adj_mat = sp.load_npz(path + '/s_adj_mat.npz')
        norm_adj_mat = sp.load_npz(path + '/s_norm_adj_mat.npz')
        mean_adj_mat = sp.load_npz(path + '/s_mean_adj_mat.npz')
        print('already load adj matrix', adj_mat.shape, time.time() - t0)
    except Exception:
        adj_mat, norm_adj_mat, mean_adj_mat = buildLaplacianMat(rt, userNum, itemNum)
        sp.save_npz(path + '/s_adj_mat.npz', adj_mat)
        sp.save_npz(path + '/s_norm_adj_mat.npz', norm_adj_mat)
        sp.save_npz(path + '/s_mean_adj_mat.npz', mean_adj_mat)
    adj_map['plain_adj'] = adj_mat
    adj_map['norm_adj'] = norm_adj_mat
    adj_map['mean_adj'] = mean_adj_mat
    adj_map['plain_adj'] = scipySP_torchSP(adj_mat)
    adj_map['norm_adj'] = scipySP_torchSP(norm_adj_mat)
    adj_map['mean_adj'] = scipySP_torchSP(mean_adj_mat)
    return adj_map

def buildLaplacianMat(rt, userNum, itemNum):

    rt_item = rt['itemId'] + userNum
    uiMat = coo_matrix((rt['rating'], (rt['userId'], rt['itemId'])))
    # print('uiMat shape', uiMat.shape)
    uiMat_upperPart = coo_matrix((rt['rating'], (rt['userId'], rt_item)))
    # print('uiMat_upperPart shape', uiMat_upperPart.shape)
    uiMat = uiMat.transpose()
    uiMat.resize((itemNum, userNum + itemNum))

    adj = sparse.vstack([uiMat_upperPart,uiMat])

    selfLoop = sparse.eye(userNum + itemNum)
    # def normalize_adj(adj):

    #     sumArr = (adj>0).sum(axis=1)
    #     diag = list(np.array(sumArr.flatten())[0])
    #     diag = np.power(diag,-0.5)
    #     D = sparse.diags(diag)
    #     L = D * adj * D
    #     L = sparse.coo_matrix(L)
    #     return L

    def normalize_adj(adj):
        degree = adj.sum(axis=1)
        degree = np.array(degree).reshape(-1)
        d_inv_sqrt = np.power(degree,-0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D = sp.diags(d_inv_sqrt)
        L = D.dot(adj).dot(D)
        L = sparse.coo_matrix(L)
        return L
    # A' = (D + I)^-1/2  ( A + I )  (D + I)^-1/2
    norm_adj = normalize_adj(adj + selfLoop)
    # A'' = D^-1/2 A D^-1/2
    mean_adj = normalize_adj(adj)
    return adj, norm_adj, mean_adj


def check_adj_if_equal(adj):
    A = np.array(adj.todense())
    degree = np.sum(A, axis=1, keepdims=False)
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D = np.diag(d_inv_sqrt)
    temp = D.dot(A).dot(D)
    # temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
    print('check normalized adjacency matrix whether equal to this laplacian matrix.')
    return temp

def scipySP_torchSP(L):
    idx = torch.LongTensor([L.row, L.col])
    data = torch.FloatTensor(L.data)
    return torch.sparse.FloatTensor(idx, data)





        
def split_data(df, dataset, ratio_train):
    if dataset == "Amazon":
        train_len =int(len(df) * ratio_train)
        train_df = df.iloc[:train_len, ]
        test_df = df.iloc[train_len:, ]
    else:
        train_df, test_df = split_loo(df)
    return train_df, test_df 

# load_data: prepare data with negative sampling for training and test
def load_data_negsample(df, dataset, ratio_train):

    df = rating_process(df)
    train_df, test_df = split_data(df, dataset, ratio_train)
    print("训练集数目：", len(train_df))
    print("测试集数目：", len(test_df))
    negatives = negtive_sampler(df)

    train_data = construct_data(train_df, negatives, 4)
    test_data = construct_data(test_df, negatives, 99)
    print("训练集数目（after negative sampling）",train_data.shape)
    print("测试集数目（after negative sampling）",test_data.shape)

    # print(train_data)
    return train_data, test_data


# rating_process: binarizing or normalizing the column rating
def rating_process(datadf, binarize=True):
    ratings = deepcopy(datadf)
    # normalize or binarize
    if binarize:
        # binarize into 0 or 1, implicit feedback
        # 就是把之前的rating列，全部换成1
        ratings['rating'][ratings['rating'] > 0] = 1.0
    else:
        # normalize into [0, 1] from [0, max_rating], explicit feedback
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
    return ratings

  

# split_loo: split into train_data and test_data according timestamp
def split_loo(datadf):
    """leave one out train/test split """
    datadf['rank_latest'] = datadf.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    test = datadf[datadf['rank_latest'] == 1]
    train = datadf[datadf['rank_latest'] > 1]
    assert train['userId'].nunique() == test['userId'].nunique()
    return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

# negtive_sampler: 
def negtive_sampler(datadf):
    """
    args:
        ratings: pd.DataFrame, which contains 3 columns = ['userId', 'itemId', 'rating']
    """
    user_pool = set(datadf['userId'].unique())  # user_pool: 943
    item_pool = set(datadf['itemId'].unique())  # item_pool: 1682

    interact_status = datadf.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
    # print(interact_status.head(5))
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
    # interact_status['snegative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
    # return interact_status[['userId', 'negative_items', 'negative_samples']]
    return interact_status[['userId', 'negative_items']]

# construct_data: 
def construct_data(df, negatives, n_neg):
    users, items, ratings = [], [], []
    data_ratings = pd.merge(df, negatives[['userId', 'negative_items']], on='userId')
    data_ratings['negatives'] = data_ratings['negative_items'].apply(lambda x: random.sample(x, n_neg))
    for row in data_ratings.itertuples():
        users.append(int(row.userId))
        items.append(int(row.itemId))
        ratings.append(float(row.rating))
        for i in range(n_neg):
            users.append(int(row.userId))
            items.append(int(row.negatives[i]))
            ratings.append(float(0))  # negative samples get 0 rating
        
    return np.stack([users, items, ratings], axis=1)

if __name__ == "__main__":
    # # check ml100k laplacian
    # train_data, test_data, userNum, itemNum, adj_map = load_data('ml100k', 'RANK', 0.7)
    # mean_adj = check_adj_if_equal(adj_map['plain_adj'])
    # print(np.where(mean_adj != adj_map['mean_adj'].todense()))

    # # check ml1m laplacian
    # train_data, test_data, userNum, itemNum, adj_map = load_data('ml1m', 'MSE', 0.7)
    # mean_adj = check_adj_if_equal(adj_map['plain_adj'])
    # print(np.where(mean_adj != adj_map['mean_adj'].todense()))

    # check Amazon laplacian
    train_data, test_data, userNum, itemNum, adj_map = load_data('Amazon', 'BPR', 0.7)
    mean_adj = check_adj_if_equal(adj_map['plain_adj'])
    print(np.where(mean_adj != adj_map['mean_adj'].todense()))