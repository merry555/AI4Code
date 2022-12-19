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
from utils.preprocess import get_features, generate_triplet

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
from models.model import MarkdownModel, MarkdownRDModel, PairwiseModel
from valid import validate

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.filterwarnings(action='ignore')

import torch, gc
gc.collect()
torch.cuda.empty_cache()


def read_data(data): return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

def validate(model, val_loader, mode='train'):
    model.eval()
    
    tbar = tqdm(val_loader, file=sys.stdout)
    
    preds = np.zeros(len(val_loader.dataset), dtype='float32')
    labels = []
    count = 0

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1]).detach().cpu().numpy().ravel()

            preds[count:count+len(pred)] = pred
            count += len(pred)
            
            if mode=='test':
              labels.append(target.detach().cpu().numpy().ravel())
    if mode=='test':
      return preds
    else:
      return np.concatenate(labels), np.concatenate(preds)

    
def predict(test_df, model, ckpt_path, test_loader, submission_name):
    model = model.cuda()
    model.eval()
    model.load_state_dict(torch.load(ckpt_path))
    _, y_test = validate(model, test_loader)
    model.to(torch.device('cpu'))
    torch.cuda.empty_cache()    
    del model, test_loader
    gc.collect()      
    
    test_df.loc[test_df["cell_type"] == "markdown", "pred"] = y_test
    sub_df = test_df.sort_values("pred").groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
    sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
    sub_df.head()
    sub_df.to_csv(f"{submission_name}.csv", index=False)

    del test_df, ckpt_path, sub_df
    gc.collect()




if __name__ == '__main__':
    data_dir = Path('../input/AI4Code')

    paths_test = list((data_dir / 'test').glob('*.json'))

    notebooks_test = [
        read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
    ]

    test_df = (
        pd.concat(notebooks_test)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    ).reset_index()
    test_df["rank"] = test_df.groupby(["id", "cell_type"]).cumcount()
    test_df["pred"] = test_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    test_df["pct_rank"] = 0
    test_fts = get_features(test_df)


    ckpt_path1 = 'graphcodebert_alldata_24_64_512_5_model_4epochs.bin' ## graph code bert path
    ckpt_path2 = 'fulldata__finetuned_graphcodebert_lr3e-05_rdrop_a05_pytorch_model_4epoch.bin'
    ckpt_path3 = 'graphcodebert-24-64-512-5-model-5epochs.bin'

    model_path = "../input/graphcodebertbase/graphcodebert-base"

    
    test_ds = MarkdownDataset(test_df[test_df["cell_type"] == "markdown"].reset_index(drop=True), md_max_len=TEST_CFG['md_max_len'],total_max_len=TEST_CFG['total_max_len'], model_name_or_path=model_path, fts=test_fts)
    test_loader = DataLoader(test_ds, batch_size=TEST_CFG['batch_size'], shuffle=False, num_workers=TEST_CFG['n_workers'],
                              pin_memory=False, drop_last=False)

    #### only graphcodebert 4epoch
    model = MarkdownModel(model_path)
    predict(test_df, model, ckpt_path1, test_loader, 'submission_1') 
    del model

    #### graphcodebert + r_droupout
    model = MarkdownRDModel(model_path)
    predict(test_df, model, ckpt_path2, test_loader, 'submission_2')
    del model

    #### graphcodebert 5epoch
    model = MarkdownModel(model_path)
    predict(test_df, model, ckpt_path3, test_loader, 'submission_3')
    del model


    #### pairwise
    test_triplets = generate_triplet(test_df, mode = 'test')
    test_ds = PairwiseDataset(test_triplets, max_len=TEST_CFG['total_max_len'])
    test_loader = DataLoader(test_ds, batch_size=TEST_CFG['batch_size'] * 4, shuffle=False, num_workers=NW, pin_memory=False, drop_last=False)

    model = PairwiseModel()
    model = model.cuda()
    model.load_state_dict(torch.load('my_own_model.bin'))
    y_test = validate(model, test_loader, mode='test')

    preds_copy = y_test
    pred_vals = []
    count = 0

    for id, df_tmp in tqdm(test_df.groupby('id')):
        df_tmp_mark = df_tmp[df_tmp['cell_type']=='markdown']
        df_tmp_code = df_tmp[df_tmp['cell_type']!='markdown']
        df_tmp_code_rank = df_tmp_code['rank'].rank().values
        N_code = len(df_tmp_code_rank)
        N_mark = len(df_tmp_mark)
        preds_tmp = preds_copy[count:count+N_mark * N_code]

        count += N_mark * N_code

        for i in range(N_mark):
            pred = preds_tmp[i*N_code:i*N_code+N_code] 

            softmax = np.exp((pred-np.mean(pred)) *20)/np.sum(np.exp((pred-np.mean(pred)) *20)) 

            rank = np.sum(softmax * df_tmp_code_rank)
            pred_vals.append(rank)  

    del model
    del test_triplets[:]
    del dict_cellid_source
    gc.collect()

    test_df.loc[test_df["cell_type"] == "markdown", "pred"] = pred_vals
    sub_df = test_df.sort_values("pred").groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
    sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
    sub_df.to_csv("submission_4.csv", index=False)


    #### Rank ensemble
    df_1 = pd.read_csv('submission_3.csv') 
    df_3 = pd.read_csv('submission_1.csv') 
    df_4 = pd.read_csv('submission_4.csv') 
    df_2 = pd.read_csv('submission_2.csv') 

    new_samples = []
    for sample_idx in range(len(df_1)):
        sample_1 = {k: v for v, k in enumerate(df_1.iloc[sample_idx]['cell_order'].split(' '))}
        sample_3 = {k: v for v, k in enumerate(df_3.iloc[sample_idx]['cell_order'].split(' '))}
        sample_4 = {k: v for v, k in enumerate(df_4.iloc[sample_idx]['cell_order'].split(' '))}
        for key in sample_1: sample_1[key] = ( (sample_1[key] * 0.333) + (sample_3[key] * 0.334) + (sample_4[key] * 0.333))
        new_samples.append(' '.join([i[0] for i in list(sorted(sample_1.items(), key=lambda x:x[1]))]))
    df_1['cell_order'] = new_samples

    new_samples = []
    for sample_idx in range(len(df_1)):
        sample_1 = {k: v for v, k in enumerate(df_1.iloc[sample_idx]['cell_order'].split(' '))}
        sample_2 = {k: v for v, k in enumerate(df_2.iloc[sample_idx]['cell_order'].split(' '))}
        for key in sample_1: sample_1[key] = ( (sample_1[key] * 0.55) + (sample_2[key] * 0.45) )
        new_samples.append(' '.join([i[0] for i in list(sorted(sample_1.items(), key=lambda x:x[1]))]))
    df_1['cell_order'] = new_samples

    df_1.to_csv('submission.csv', index = False)