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

# dataset:movielens100K
def load100KRatings(datapath):
    rt = pd.read_table(datapath + '/u.data',sep='\t',names=['userId','itemId','rating','timestamp'])
    return rt

# dataset:movielens1M
def load1MRatings(datapath):
    rt = pd.read_table(datapath + '/ratings.dat',sep='::',names=['userId','itemId','rating','timestamp'])
    return rt

# dataset:Amazonbook
def loadAmazon(datapath):
    train_df = pd.read_table(datapath + '/Amazon_train.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    test_df = pd.read_table(datapath + '/Amazon_test.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    rt = train_df.append(test_df)
    return rt, train_df, test_df

def loadGowalla(datapath):
    train_df = pd.read_table(datapath + '/g_train.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    test_df = pd.read_table(datapath + '/g_test.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    rt = train_df.append(test_df)
    return rt, train_df, test_df

def loadAmazon_Gowalla(dataset):
    datapath = path.dirname(__file__) + '/' + dataset
    if dataset == 'Amazon':
        return loadAmazon(datapath)
    if dataset == 'Gowalla':
        return loadGowalla(datapath)

# 构建正负样本集合：userId：int; positive_items：set; negative_items: set.
# 负样本集合为所有未在训练集中出现过的item.
def train_positives_negtives(item_pool, train_df):
    train_df = train_df.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'positive_items'})
    train_df['negative_items'] = train_df['positive_items'].apply(lambda x: item_pool - x)
    return train_df[['userId', 'positive_items', 'negative_items']]

def sample_train_pair(train_df):
    # train_df: ['userId', 'positive_items', 'negative_items']
    sampled_batch = train_df.sample(n=len(train_df))
    # sampled_batch = train_df
    # sample pairs
    sampled_batch['pos_sample'] = sampled_batch['positive_items'].apply(lambda x: random.sample(x, 1))
    sampled_batch['neg_sample'] = sampled_batch['negative_items'].apply(lambda x: random.sample(x, 1))
    return sampled_batch[['userId', 'pos_sample', 'neg_sample']]

def test_positives_negtives(train_df, test_df):
    test_df = test_df.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'positive_items'})
    test_df = pd.merge(test_df, train_df[['userId', 'negative_items']], on='userId')
    test_user_num = len(test_df['userId'].unique())
    return test_df[['userId', 'positive_items', 'negative_items']], test_user_num

def load_data(dataset, evaluate, ratio_train, adj_type):
    if dataset in ['Amazon', 'Gowalla']:
        datapath = path.dirname(__file__) + '/' + dataset
        rt, train_df, test_df = loadAmazon_Gowalla(dataset)

        item_pool = set(rt['itemId'].unique()) 
        
        # Amazon and Gowalla index starts at 0
        userNum = rt['userId'].max() + 1  
        itemNum = rt['itemId'].max() + 1
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
        if evaluate == 'BPR' :
            t0 = time.time()
            # Generate Train_data
            train_df = train_positives_negtives(item_pool, train_df)
            train_data = sample_train_pair(train_df)
            train_data = PairDataset(train_data.values)

            # Generate Test_data
            test_df, test_user_num = test_positives_negtives(train_df, test_df)
            test_data = TestDataSet(test_df.values)
            
            print('Time consuming of generating train_test data:', time.time() - t0)
            print('test_user_num:{}'.format(test_user_num))
            print('number of trains:{}'.format(len(train_df)))

        elif evaluate == 'RANK':
            train_data, test_data = load_data_negsample(rt, dataset, ratio_train)
            train_data = MLDataSet(train_data)
            test_data = MLDataSet(test_data)

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
            item_pool = set(rt['itemId'].unique()) 
            # Generate Train_data
            train_df = train_positives_negtives(item_pool, train_df)
            train_data = sample_train_pair(train_df)
            train_data = PairDataset(train_data.values)
            # Generate Test_data
            test_df, test_user_num = test_positives_negtives(train_df, test_df)
            test_data = TestDataSet(test_df.values)
            print('test_user_num:{}'.format(test_user_num))
    
    adj = get_adj_mat(datapath, rt, userNum, itemNum, adj_type)

    return train_data, test_data, userNum, itemNum, adj, test_user_num

def scipySP_torchSP(L):
    idx = torch.LongTensor([L.row, L.col])
    data = torch.FloatTensor(L.data)
    return torch.sparse.FloatTensor(idx, data)

def buildLaplacianMat(rt, userNum, itemNum, adj_type):
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
        adj = adj.tocsr()
        degree = sparse.csr_matrix(adj.sum(axis=1))
        d_inv_sqrt = degree.power(-0.5) # csr_matrix (size ,1) 
        d_inv_sqrt = np.array(d_inv_sqrt.todense()).reshape(-1)
        D = sparse.diags(d_inv_sqrt)
        L = D.dot(adj).dot(D) # csr_matrix (size, size)
        return sparse.coo_matrix(L)
    
    if adj_type == 'plain_adj':
        return adj
    # A' = (D + I)^-1/2  ( A + I )  (D + I)^-1/2
    elif adj_type == 'norm_adj':
        return normalize_adj(adj + selfLoop)
    # A'' = D^-1/2 A D^-1/2
    elif adj_type == 'mean_adj':
        return normalize_adj(adj)

def get_adj_mat(path, rt, userNum, itemNum, adj_type):
    try:
        if adj_type == 'plain_adj':
            adj = sp.load_npz(path + '/s_adj_mat.npz')
        elif adj_type == 'norm_adj':
            adj = sp.load_npz(path + '/s_norm_adj_mat.npz')
        elif adj_type == 'mean_adj':
            adj = sp.load_npz(path + '/s_mean_adj_mat.npz')
        print('Load adj matrix', adj.shape)
    except Exception:
        t0 = time.time()
        adj = buildLaplacianMat(rt, userNum, itemNum, adj_type)
        print('Time consuming for building adj matrix', adj.shape, time.time() - t0)
        if adj_type == 'plain_adj':
            sp.save_npz(path + '/s_adj_mat.npz', adj)
        elif adj_type == 'norm_adj':
            sp.save_npz(path + '/s_norm_adj_mat.npz', adj)
        elif adj_type == 'mean_adj':  
            sp.save_npz(path + '/s_mean_adj_mat.npz', adj)
    return scipySP_torchSP(adj)

# temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
def check_adj_if_equal(adj):
    A = np.array(adj.todense())
    degree = np.sum(A, axis=1, keepdims=False)
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D = np.diag(d_inv_sqrt)
    print('check normalized adjacency matrix whether equal to this laplacian matrix.')
    return D.dot(A).dot(D)
        
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
    pass
    # datapath = path.dirname(__file__) + '/Gowalla'
    # rt, train_df, test_df = loadGowalla(datapath)
    # item_pool = set(rt['itemId'].unique()) 
    # train_df = train_positives_negtives(item_pool, train_df)
    # test_df, test_user_num = test_positives_negtives(train_df, test_df)
    # print(test_df.iloc[0])

    # # check ml100k laplacian
    # datapath = path.dirname(__file__) + '/1K'
    # rt = load100KRatings(datapath)
    
    # check ml1m laplacian
    # datapath = path.dirname(__file__) + '/1M'
    # rt = load1MRatings(datapath)
    # userNum = rt['userId'].max()
    # itemNum = rt['itemId'].max()
    # rt['userId'] = rt['userId'] - 1
    # rt['itemId'] = rt['itemId'] - 1
    # adj, norm_adj, mean_adj = buildLaplacianMat(rt, userNum, itemNum)

    # dense_mean_adj = check_adj_if_equal(adj)
    # print(np.where(dense_mean_adj != mean_adj.todense()))

    # # check ml1m laplacian
    # train_data, test_data, userNum, itemNum, adj_map = load_data('ml1m', 'MSE', 0.7)
    # mean_adj = check_adj_if_equal(adj_map['plain_adj'])
    # print(np.where(mean_adj != adj_map['mean_adj'].todense()))

    # datapath = path.dirname(__file__) + '/Amazon'
    # rt, train_df, test_df = loadAmazon(datapath)

    # userNum = rt['userId'].max()
    # itemNum = rt['itemId'].max()
    # print('userNum:{}, itemNum:{}'.format(userNum, itemNum))

    # check Amazon laplacian
    # train_data, test_data, userNum, itemNum, adj_map = load_data('Amazon', 'BPR', 0.7)
    # mean_adj = check_adj_if_equal(adj_map['plain_adj'])
    # print(np.where(mean_adj != adj_map['mean_adj'].todense()))