import torch

PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
CONFIG_PATH = 'config.py'
DEBUG_MODE = False

class DataConfig():
    train_data_path = './data/train_dataset_v2.tsv'
    test_data_path = './data/test_dataset.tsv'
    max_sentence_lenth = 64
    bert_model_name = PRE_TRAINED_MODEL_NAME
    debug_mode = DEBUG_MODE

class ModelConfig():
    bert_model_name = PRE_TRAINED_MODEL_NAME
    linear_dropout_rate = 0.5

class TrainingConfig():
    output_dir = 'output'
    cuda_idx = 1
    train_device = torch.device('cuda:' + str(cuda_idx))
    batch_size = 64
    epoch = 10

    debug_mode = DEBUG_MODE
    save_score_limit = 0.69

    weight_decay=1e-3
    betas = (0.9, 0.999)
    eps=1e-08
    Bert_lr = 1e-5
    lr = 3e-4
    skip_loss = 0

    warm_up_ratio = 0
    warmup_proportion=0.0