TRAIN_CFG = {
    'md_max_len':64,
    'total_max_len':512,
    'batch_size':8,
    'accumulation_steps':  4,
    'kl_alpha': 0.5,
    'epochs':4,
    'learning_rate':3e-5,
    'n_workers':8,
    'model_path':'./model/pretrain/graphcodebert-base',
    'r_dropout':True
}

TEST_CFG = {
    'md_max_len':64,
    'total_max_len':512,
    'batch_size':32,
    'n_workers':8,
}