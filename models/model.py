from tqdm import tqdm
import sys, os
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn 
import torch

class MarkdownModel(nn.Module):
        def __init__(self, model_path):
            super(MarkdownModel, self).__init__()
            self.model = AutoModel.from_pretrained(model_path)
            self.top = nn.Linear(769, 1)

        def forward(self, ids, mask, fts):
            x = self.model(ids, mask)[0]
            x = self.top(torch.cat((x[:, 0, :], fts),1))
            return x
 
class MarkdownRDModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.ReLU(),
        )
        self.top = nn.Linear(768+1, 1) #768 + ft(=1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0] # 이게 hidden_states부분
        x = self.net(x[:, 0, :]) # r_dropout layer
        x = self.top(torch.cat((x, fts),1)) # 3차 수정
        return x


class PairwiseModel(nn.Module):
    def __init__(self, model_path):
        super(PairwiseModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x) 
        return x