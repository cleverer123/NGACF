import numpy as np
import pandas as pd
from os import path
import torch


path = path.dirname(__file__)

data = []
with open(path + '/test.txt', 'r') as fd:
    line = fd.readline()
    while line != None and line != '' :
        arr = line.split( )
        u = eval(arr[0])
        for i in arr[1:]:
            data.append([u, int(i), int(1)])
        line = fd.readline()

print(np.array(data).shape)
data_df = pd.DataFrame(data)
data_df.to_csv('y_test.csv', header=False, index=False)

# df = pd.read_table(pathAmazon + '/aaaaa.csv',sep=',', names=['userId','itemId','rating'])
# print(df.iloc[20:30,])