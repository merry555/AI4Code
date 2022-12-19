import random
import pandas as pd
import numpy as np
import itertools

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.loss import compute_kl_loss
from utils.metrics import kendall_tau
from utils.preprocess import generate_triplet

from tqdm.auto import tqdm
import sys

import os
import torch
import gc
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

#### import files
from config import TRAIN_CFG
from dataset import MarkdownDataset, PairwiseDataset ## image + text
from models.model import MarkdownModel, PairwiseModel

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.filterwarnings(action='ignore')

import torch, gc
gc.collect()
torch.cuda.empty_cache()

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


# def validate(model, val_loader):
#     model.eval()

#     tbar = tqdm(val_loader, file=sys.stdout)

#     preds = []
#     labels = []

#     with torch.no_grad():
#         for idx, data in enumerate(tbar):
#             inputs, target = read_data(data)

#             with torch.cuda.amp.autocast():
#                 pred = model(*inputs)

#             preds.append(pred.detach().cpu().numpy().ravel())
#             labels.append(target.detach().cpu().numpy().ravel())

#     return np.concatenate(labels), np.concatenate(preds)

def train(model, train_loader, epochs):
    np.random.seed(0)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(epochs * len(train_loader) / accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=TRAIN_CFG['learning_rate'],
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler


    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    
    for e in range(epochs):   
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
                
        loss_list = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1: #파라미터 찾으려고 스케줄러 실행해주는데
                                        # 매스텝마다 실행시키면 시간이 오래걸려서, 일정 조건을 주고 하도록 ex) 4번에 1번
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            avg_loss = np.round(np.mean(loss_list), 4)
            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss}")
        
        output_model_file = f"./my_own_model_{e}.bin"
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == '__main__':
    gc.collect()

    model_path = 'mymodelbertsmallpretrained/checkpoint-120000'

    df = pd.read_csv('pairwise_dataset.csv')
    triplets = generate_triplet(df)

    train_ds = PairwiseDataset(triplets, max_len=MAX_LEN, mode='train')
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CFG['batch_size'], shuffle=True, num_workers=TRAIN_CFG['num_workers'],
                          pin_memory=False, drop_last=True)

    model = PairwiseModel()
    model = model.cuda()


    model = train(model, train_loader, epochs=TRAIN_CFG['epochs'])
    gc.collect()
