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
# from sklearn.model_selection import train_test_split

from data.mldataset import MLDataSet, PairDataset, AllNegtivesDataSet, SampledNegtivesDataSet

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

def loadYelp(datapath):
    train_df = pd.read_table(datapath + '/y_train.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    test_df = pd.read_table(datapath + '/y_test.csv',sep=',', names=['userId','itemId','rating'], dtype={'userId': np.int64, 'itemId': np.int64})
    rt = train_df.append(test_df)
    return rt, train_df, test_df

def loadAmazon_Gowalla(dataset):
    datapath = path.dirname(__file__) + '/' + dataset
    if dataset == 'Amazon':
        return loadAmazon(datapath)
    if dataset == 'Gowalla':
        return loadGowalla(datapath)

# 构建正负样本集合：userId：int; positive_items：set; negative_items: set.
# 负样本集合为所有未在训练集、测试集中出现过的item, 即 item_pool - trian_items - test_items
def positives_negtives(rt):
    item_pool = set(rt['itemId'].unique()) 
    pos_neg = rt.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'positive_items'})
    pos_neg['negative_items'] = pos_neg['positive_items'].apply(lambda x: item_pool - x)
    return pos_neg[['userId', 'positive_items', 'negative_items']]

# 对于trainMode == 'PairSampling', 负样本为item_pool - trian_items 
def train_pos_neg_exclude_test(rt, train_df):
    item_pool = set(rt['itemId'].unique()) 
    train_pos_neg = train_df.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'positive_items'})
    train_pos_neg['negative_items'] = train_pos_neg['positive_items'].apply(lambda x: item_pool - x)
    return train_pos_neg[['userId', 'positive_items', 'negative_items']]

def train_pair_sampling(train_df, pos_neg):
    # train_df: ['userId', 'positive_items', 'negative_items']
    # sampled_batch = train_pos_neg.sample(n=len(train_pos_neg))
    # sampled_batch = train_df
    # sample pairs
    sampled_train_pair = pd.merge(train_df, pos_neg[['userId', 'positive_items', 'negative_items']], on='userId')
    sampled_train_pair['pos_sample'] = sampled_train_pair['positive_items'].apply(lambda x: random.sample(x, 1))
    sampled_train_pair['neg_sample'] = sampled_train_pair['negative_items'].apply(lambda x: random.sample(x, 1))
    return sampled_train_pair[['userId', 'pos_sample', 'neg_sample']]

# 对于 trainMode == 'NegSampling', 负样本为 item_pool - trian_items - test_items， 同pos_neg
def train_neg_sampling(train_df, pos_neg):
    sampled_train_neg = pd.merge(train_df, pos_neg[['userId', 'negative_items']], on='userId')
    sampled_train_neg['neg_samples'] = sampled_train_neg['negative_items'].apply(lambda x: random.sample(x, 4))
    return sampled_train_neg[['userId', 'itemId', 'neg_samples']]


def test_positives(test_df):
    test_pos = test_df.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'positive_items'})
    return test_pos[['userId', 'positive_items']]

def test_positives_negtives(test_pos, train_pos_neg):
    test_pos_neg = pd.merge(test_pos, train_pos_neg[['userId', 'negative_items']], on='userId')
    return test_pos_neg[['userId', 'positive_items', 'negative_items']]

#  # 对于 evalMode == 'AllNeg', 负样本为 item_pool - trian_items
# def test_positives_negtives(test_df, train_pos_neg):
#     test_pos_neg = test_df.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'positive_items'})
#     test_pos_neg = pd.merge(test_pos_neg, train_pos_neg[['userId', 'negative_items']], on='userId')
#     test_user_num = len(test_pos_neg['userId'].unique())
#     return test_pos_neg[['userId', 'positive_items', 'negative_items']], test_user_num

def test_neg_sampling(test_df, pos_neg):
    sample_test_neg = pd.merge(test_df, pos_neg[['userId', 'negative_items']], on='userId')
    sample_test_neg['neg_samples'] = sample_test_neg['negative_items'].apply(lambda x: random.sample(x, 99))
    test_user_num = len(sample_test_neg['userId'].unique())
    return sample_test_neg[['userId', 'itemId', 'neg_samples']], test_user_num

def load_train_test_data(rt, train_df, test_df, trainMode, evalmode):
    pos_neg = positives_negtives(rt)
    print('-----------------All data. pos_neg ----------------- \n',pos_neg.head(3))
    # Generate train_data
    train_pos_neg = train_pos_neg_exclude_test(rt, train_df)
    if trainMode == 'PairSampling': # 一行 代表 user的 一个 pos_item 和 一个 neg_item 
        sampled_train_pair = train_pair_sampling(train_df, train_pos_neg)
        print('-----------------trainMode == PairSampling. sampled_train_pair ----------------- \n', sampled_train_pair.head(3))
        train_data = PairDataset(sampled_train_pair.values)          
    elif trainMode == 'NegSampling': # 一行 代表 user的 一个 pos_item 和 所有 neg_items
        sampled_train_neg = train_neg_sampling(train_df, pos_neg)
        # neg_sample_with_label = construct_neg_samples_labels(sampled_train_neg)
        # train_data = MLDataSet(neg_sample_with_label)
        print('-----------------trainMode == NegSampling. sampled_train_neg ----------------- \n', sampled_train_neg.head(3))
        train_data = SampledNegtivesDataSet(sampled_train_neg.values)
    # Generate test_data
    if evalmode =='AllNeg': # 一行 代表 user的 所有 pos_items 和 所有 neg_items
        test_pos_neg, test_user_num = test_positives_negtives(test_df, train_pos_neg)   
        print('-----------------evalmode ==AllNeg. test_pos_neg: -----------------\n', test_pos_neg.head(3)) 
        # test_data = AllNegtivesDataSet(test_pos_neg.values)
        test_data = test_pos_neg
    elif evalmode == 'SampledNeg': # 一行 代表 user的 一个 pos_item 和 所有 neg_items
        sampled_test_neg, test_user_num = test_neg_sampling(test_df, pos_neg)
        print('-----------------evalmode == SampledNeg. sampled_test_neg ----------------- \n', sampled_test_neg.head(3))
        test_data = SampledNegtivesDataSet(sampled_test_neg.values)
    print('Size of train_data:{} test_data:{}'.format(len(train_data), len(test_data)))
    return train_data, test_data, test_user_num
    
def load_data_adj(dataset, ratio_train, adj_type, trainMode, evalmode):
    if dataset in ['Amazon', 'Gowalla']:
        datapath = path.dirname(__file__) + '/' + dataset
        rt, train_df, test_df = loadAmazon_Gowalla(dataset)
        # Amazon and Gowalla index starts at 0
        userNum = rt['userId'].max() + 1  
        itemNum = rt['itemId'].max() + 1
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
    elif dataset == 'ml1m':
        datapath = path.dirname(__file__) + '/1M'
        rt = load1MRatings(datapath)
        userNum = rt['userId'].max()
        itemNum = rt['itemId'].max()
        rt['userId'] = rt['userId'] - 1
        rt['itemId'] = rt['itemId'] - 1
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
        # # split random
        # train_df, test_df = loadML1m()
        # train_df['userId'] = train_df['userId'] - 1
        # train_df['itemId'] = train_df['itemId'] - 1
        # test_df['userId'] = test_df['userId'] - 1
        # test_df['itemId'] = test_df['itemId'] - 1
        
        # # split by leave one out  
        train_df, test_df = split_loo(rt)

    elif dataset == 'ml100k':
        datapath = path.dirname(__file__) + '/1K'
        rt = load100KRatings(datapath)
        # ml100k index starts at 1
        userNum = rt['userId'].max()
        itemNum = rt['itemId'].max()
        print('userNum:{}, itemNum:{}'.format(userNum, itemNum))
        rt['userId'] = rt['userId'] - 1
        rt['itemId'] = rt['itemId'] - 1
        train_df, test_df = split_loo(rt)
    print('train_len:{}, test_len:{}'.format(len(train_df), len(test_df)))

    t0 = time.time()
    train_data, test_data, test_user_num = load_train_test_data(rt, train_df, test_df, trainMode, evalmode)
    print('Time consuming of generating train_test data:', time.time() - t0)
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
    uiMat_lowerPart = uiMat.transpose()
    uiMat_lowerPart.resize((itemNum, userNum + itemNum))

    adj = sparse.vstack([uiMat_upperPart,uiMat_lowerPart])

    selfLoop = sparse.eye(userNum + itemNum)

    def normalize_adj(adj):
        adj = adj.tocsr()
        degree = sparse.csr_matrix(adj.sum(axis=1))
        d_inv_sqrt = degree.power(-0.5) # csr_matrix (size ,1) 
        d_inv_sqrt = np.array(d_inv_sqrt.todense()).reshape(-1)
        D = sparse.diags(d_inv_sqrt)
        L = D.dot(adj).dot(D) # csr_matrix (size, size)
        return sparse.coo_matrix(L)

    # def normalize_adj(adj):
    #     adj = adj.tocsr()
    #     degree = sparse.csr_matrix(adj.sum(axis=1))
    #     degree = np.array(degree.todense())
    #     d_inv_sqrt = 1.0/degree 
    #     d_inv_sqrt = d_inv_sqrt.reshape(-1)
    #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    #     D = sparse.diags(d_inv_sqrt)
    #     L = D.dot(adj) # csr_matrix (size, size)
    #     return sparse.coo_matrix(L)
    
    # A' = (D + I)^-1/2  ( A + I )  (D + I)^-1/2
    if adj_type == 'ui_mat':
        return uiMat
    elif adj_type == 'plain_adj':
        return adj
    elif adj_type == 'norm_adj':
        return normalize_adj(adj + selfLoop)
    # A'' = D^-1/2 A D^-1/2
    elif adj_type == 'mean_adj':
        # return (adj + selfLoop).tocoo(), normalize_adj(adj)
        return normalize_adj(adj)

def get_adj_mat(path, rt, userNum, itemNum, adj_type):
    try:
        if adj_type == 'ui_mat':
            adj = sp.load_npz(path + '/s_ui_adj.npz')
        elif adj_type == 'plain_adj':
            adj = sp.load_npz(path + '/s_plain_adj.npz')
        if adj_type == 'norm_adj':
            adj = sp.load_npz(path + '/s_norm_adj.npz')
        elif adj_type == 'mean_adj':
            adj = sp.load_npz(path + '/s_mean_adj.npz')
        print('Load adj matrix', adj.shape)
    except Exception:
        t0 = time.time()
        adj = buildLaplacianMat(rt, userNum, itemNum, adj_type)
        print('Time consuming for building adj matrix', adj.shape, time.time() - t0)
        if adj_type == 'ui_mat':
            sp.save_npz(path + '/s_ui_adj.npz', adj)
        elif adj_type == 'plain_adj':
            sp.save_npz(path + '/s_plain_adj.npz', adj)
        elif adj_type == 'norm_adj':
            sp.save_npz(path + '/s_norm_adj.npz', adj)
        elif adj_type == 'mean_adj':  
            sp.save_npz(path + '/s_mean_adj.npz', adj)
    # return torch.from_numpy(adj.todense()), scipySP_torchSP(laplacianMat)
    return scipySP_torchSP(adj).coalesce()

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