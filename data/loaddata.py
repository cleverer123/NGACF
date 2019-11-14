import pandas as pd
from os import path

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