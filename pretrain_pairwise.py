from tqdm import tqdm
import sys, os
from transformers import DistilBertModel, DistilBertTokenizer, BertTokenizer, LineByLineTextDataset
import torch.nn.functional as F
import torch.nn as nn
import torch

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel

import tokenizers
from utils.preprocess import preprocess_text, read_notebook

NUM_TRAIN = 200


if __name__ == '__main__':
    data_dir = Path('./data/ai4code')

    paths_train = list((data_dir / 'train').glob('*.json'))[:NUM_TRAIN]
    notebooks_train = [
        read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
    ]
    
    df = (
        pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    )

    df_orders_ = df_orders.to_frame().join(
        df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
        how='right',
    )


    ranks = {}
    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

    df_ranks = (
        pd.DataFrame
        .from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
    )


    df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
    df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
    df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

    df.source = df.source.apply(preprocess_text)

    df.to_csv('pairwise_dataset.csv', index=False)

    if not os.path.exists('text.txt'):
        with open('text.txt','w') as f:
            for id, item in tqdm(df.groupby('id')):
                df_markdown =  item[item['cell_type']=='markdown']
                for source, rank in df_markdown[['source', 'rank']].values:
                    cell_source = df_markdown[df_markdown['rank']==(rank+1)]
                    if len(cell_source):
                    setence = source + ' [SEP] ' + cell_source.source.values[0]
                    f.write(setence+'\n')    

    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

    model = AutoModelWithLMHead.from_pretrained('prajjwal1/bert-small')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    from transformers import Trainer, TrainingArguments

    dataset= LineByLineTextDataset(
        tokenizer = tokenizer,
        file_path = './text.txt',
        block_size = 128  # maximum sequence length
    )

    print('No. of lines: ', len(dataset)) # No of lines in your datset

    training_args = TrainingArguments(
        output_dir='./',
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=64,
        save_steps=10000,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model('./')