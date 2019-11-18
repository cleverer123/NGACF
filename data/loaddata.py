import os
from os import path
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import random_split
import scipy.sparse as sp
from scipy.sparse.coo import coo_matrix 

# load 100k data
# path100k = path.dirname(__file__) + r'\1K'
path100k = path.dirname(__file__) + '/1K'

def load100KRatings():
    # df = pd.read_table(path100k+r'\u.data',sep='\t',names=['userId','itemId','rating','timestamp'])
    df = pd.read_table(path100k + '/u.data',sep='\t',names=['userId','itemId','rating','timestamp'])
    return df

path1M = path.dirname(__file__) + '/1M'
def load1MRatings():
    # df = pd.read_table(path100k+r'\u.data',sep='\t',names=['userId','itemId','rating','timestamp'])
    df = pd.read_table(path1M + '/ratings.dat',sep='::',names=['userId','itemId','rating','timestamp'])
    return df

def load100KItemSide():
    import codecs
    with codecs.open(path100k+'/u.item', 'r', 'utf-8', errors='ignore') as f:
        movies = pd.read_table(f, delimiter='|', header=None,names="itemId| movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western ".split('|'))
    return movies

def load100kUserSide():
    import codecs
    with codecs.open(path100k + '/u.user', 'r', 'utf-8', errors='ignore') as f:
        users = pd.read_table(f, delimiter='|', header=None,names="userId| age | gender | occupation | zip code".split('|'))
    return users


# load_data: prepare data with negative sampling for training and test
def load_data(df, ratio_train):

    df = rating_process(df)
    train_df, test_df = split_loo(df)
    print("训练集数目：", len(train_df))
    print("测试集数目：", len(test_df))
    negatives = negtive_sampler(df)

    train_data = construct_data(train_df, negatives, 4)
    test_data = construct_data(test_df, negatives, 99)
    print("训练集数目（after negative sampling）",train_data.shape)
    print("测试集数目（after negative sampling）",test_data.shape)

    print(train_data)
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