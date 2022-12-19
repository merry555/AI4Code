import re
import torch
import pandas as pd
import numpy as np
import pandas as pd
import os
import re
# import fasttext
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import nltk
nltk.download('wordnet')

stemmer = WordNetLemmatizer()

def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )

def clean_code(cell): return str(cell).replace("\\n", "\n")

def sample_cells(cells, n):
    import numpy as np
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells): return [cell[:200] for cell in cells]
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step        
        if cells[-1] not in results: results[-1] = cells[-1]
        return results

def get_features(df):
    from tqdm import tqdm
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
        features[idx]["codes"] = codes
    return features

## HTML TAG 제거
def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """
    try:
        # md -> html -> text since BeautifulSoup can extract text cleanly
        html = markdown(markdown_string)

        # remove code snippets
        html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
        html = re.sub(r'<code>(.*?)</code >', ' ', html)

        # extract text
        soup = BeautifulSoup(html, "html.parser")
        text = ''.join(soup.findAll(text=True))
    
    except:
        text = markdown_string
    return text

## markdown 특수문자 제거
def clean_text(inputString):
    try:
        text_rmv = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', inputString)
    except:
        text_rmv = inputString
    return text_rmv


def preprocess_text(document):
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    
    return document

def preprocess_df(df):
    """
    This function is for processing sorce of notebook
    returns preprocessed dataframe
    """
    return [preprocess_text(message) for message in df.source]

def generate_triplet(df, mode='train'):
    triplets = []
    ids = df.id.unique()
    random_drop = np.random.random(size=10000)>0.9
    count = 0

    for id, df_tmp in tqdm(df.groupby('id')):
    df_tmp_markdown = df_tmp[df_tmp['cell_type']=='markdown']

    df_tmp_code = df_tmp[df_tmp['cell_type']=='code']
    df_tmp_code_rank = df_tmp_code['rank'].values
    df_tmp_code_cell_id = df_tmp_code['cell_id'].values

    for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
        labels = np.array([(r==(rank+1)) for r in df_tmp_code_rank]).astype('int')

        for cid, label in zip(df_tmp_code_cell_id, labels):
        count += 1
        if label==1:
            triplets.append( [cell_id, cid, label] )
            # triplets.append( [cid, cell_id, label] )
        elif mode == 'test':
            triplets.append( [cell_id, cid, label] )
            # triplets.append( [cid, cell_id, label] )
        elif random_drop[count%10000]:
            triplets.append( [cell_id, cid, label] )
            # triplets.append( [cid, cell_id, label] )

    return triplets