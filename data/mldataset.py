from torch.utils.data import Dataset
import numpy as np
class MLDataSet(Dataset):

    def __init__(self,rt):
        super(MLDataSet,self).__init__()
        # self.uId = list(rt['userId'])
        # self.iId = list(rt['itemId'])
        # self.rt = list(rt['rating'])

        self.uId = list(rt[:,0])
        self.iId = list(rt[:,1])
        self.rt = list(rt[:,2])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item])

# PairDataset is used when train bpr loss is used. 
class PairDataset(Dataset):
    def __init__(self, df):
        super(PairDataset, self).__init__()
        self.data = df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # userId, pos_itemId, neg_itemId
        return (self.data[index, 0], self.data[index, 1][0], self.data[index, 2][0])

# AllNegtivesDataSet is used when evaluate with all negtive items 
class AllNegtivesDataSet(Dataset):
    def __init__(self, data):
        super(AllNegtivesDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # test_userId, positive item set, negative item set
        return (self.data[index, 0], list(self.data[index, 1]), list(self.data[index, 2]))

# NegSampleDataSet is used when evaluate with sampled negative items
class SampledNegtivesDataSet(Dataset):
    def __init__(self, data):
        super(SampledNegtivesDataSet, self).__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # test_userId, one positive item, sampled negative item set
        return (self.data[index, 0], self.data[index, 1], list(self.data[index, 2]))

# ItemDataSet is used when evaluating test data 
class ItemDataSet(Dataset):
    def __init__(self, data):
        super(ItemDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # item
        return self.data[index]

# class TestData():
#     def __init__(self, test_df):
#         self.test_df = test_df
    
#     def getPstiveItemsByUserId(self, userId):
#         return self.test_df[self.test_df.userId==userId]['positive_items'].values
    
#     def getNegtiveTestDataByUserId(self, userId):
#         negtive_items = self.test_df[self.test_df.userId==userId]['negtive_items'].values
#         negtive_items = np.array(negtive_items).reshape(-1,1)
#         test_data = np.hstack([np.full(negtive_items.shape, userId), negtive_items])

#         return NegtiveTestDataset(test_data)
    

# class NegtiveTestDataset(Dataset):
#     def __init__(self, data):
#         super(NegtiveTestDataset, self).__init__()
#         # data : nparray
#         self.data = data
    
#     def __len__(self):
#         return data.shape[0]
    
#     def __getitem__(self, index):
#         # test_userId, neg_itemId
#         return self.data[index, 0], self.data[index, 1]