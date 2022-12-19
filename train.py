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

from tqdm.auto import tqdm
import sys

import os
import gc
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

#### import files
from config import TRAIN_CFG
from dataset import MarkdownDataset ## image + text
from models.model import MarkdownModel
from valid import validate

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.filterwarnings(action='ignore')

import torch, gc
gc.collect()
torch.cuda.empty_cache()

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

def validate(model, val_loader, mode='train'):
    model.eval()
    
    tbar = tqdm(val_loader, file=sys.stdout)
    
    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1])

            preds.append(pred.detach().cpu().numpy().ravel())
            if mode=='test':
              labels.append(target.detach().cpu().numpy().ravel())
    if mode=='test':
      return np.concatenate(preds)
    else:
      return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, val_loader, epochs, output_dir, r_dr):
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


    criterion = torch.nn.L1Loss() #L1, L2, MSE, ...
    scaler = torch.cuda.amp.GradScaler()
    
    print_idx = 0
    score_tb = pd.DataFrame() # 성능 log 기록용 테이블
    
    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            if r_dr == True:
                with torch.cuda.amp.autocast():
                    # 각 모델별 logit값 생성
                    logits = model(*inputs) # 결과값 2개 추출
                    logits2 = model(*inputs) # 결과값 2개 추출

                    # 각각의 모델 별 loss mean & kl_loss 생성
                    ce_loss = 0.5 * (criterion(logits, target) + criterion(logits2, target))
                    kl_loss = compute_kl_loss(logits, logits2)

                    loss = ce_loss + TRAIN_CFG['kl_alpha'] * kl_loss

            else:
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
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)
            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss}")

        # model save부터
        output_dir = f'/{model_save_dir}'
        os.makedirs(output_dir, exist_ok=True)
        print(f'Save model at {output_dir}/pytorch_model_e{e}.bin')
        torch.save(model.state_dict(), f"{output_dir}/pytorch_model_e{e}.bin")
        
        # Valid 진행
        if val_loader != 'None':
            y_val, y_pred = validate(model, val_loader)
            val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True) #그룹별 순위 구하기, 큰 순서대로 정렬
            val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred #마크다운엔 prediction 값
            y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
            print(f"Epoch_{e} Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
            
            # Valid 성능 진행 
            score_tb.loc[e, 'epoch'] = e
            score_tb.loc[e, 'loss'] = avg_loss
            print(f'Save_valid_result at {output_dir}')
            score_tb.to_csv(f"{output_dir}/score_history.csv", encoding = 'utf-8-sig', index=False)        

    return model

if __name__ == '__main__':
    gc.collect()

    model_name_or_path = TRAIN_CFG['model_path']

    train_df_mark = pd.read_csv(f'{data_dir}/train_mark.csv').drop("parent_id", axis=1).dropna().reset_index(drop=True)
    train_fts = json.load(open(f'{data_dir}/train_fts.json'))
    
    data_dir = Path('./data/ai4code')

    order_df = pd.read_csv(data_dir / "train_orders.csv").set_index("id")
    df_orders = pd.read_csv(
        data_dir / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()

    df_orders_cnt = df_orders.reset_index().copy()
    df_orders_cnt.loc[:,'cell_cnt'] = df_orders_cnt.cell_order.apply(lambda x: len(x)).tolist()
    del df_orders_cnt['cell_order']

    df_orders_cnt = df_orders_cnt.set_index('id')
    df_orders_dic = df_orders_cnt.to_dict()

    del df_orders_cnt
    del df_orders_dic

    train_df_mark.source = [f"{df_orders_dic['cell_cnt'][n[0]]} {n[1]}" for n in train_df_mark[['id', 'source']].values]


    train_ds = MarkdownDataset(train_df_mark, model_name_or_path=model_name_or_path, md_max_len=TRAIN_CFG['md_max_len'],
                           total_max_len=TRAIN_CFG['total_max_len'], fts=train_fts)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CFG['batch_size'], shuffle=True, num_workers=TRAIN_CFG['n_workers'],
                            pin_memory=False, drop_last=True)

    print(f">>> model import {model_name_or_path}")
    model = MarkdownModel(model_name_or_path)
    model = model.to(device)

    val_loader = 'None'
    output_model_save_dir = 'output'
    model = train(model, train_loader, val_loader
                ,epochs=TRAIN_CFG['epochs']
                ,output_dir=output_model_save_dir, TRAIN_CFG['r_dropout'])

    gc.collect()