from torch.utils.data import Dataset

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