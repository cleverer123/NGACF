import torch
from torch import nn

class BPRLoss(nn.Module):
    def __init__(self ):
        super(BPRLoss, self).__init__()
    
    def forward(self, pos_scores, neg_scores):
        return - torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
