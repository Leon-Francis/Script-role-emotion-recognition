import torch
import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim, nn
from dataset import Script_dataset
from config import TrainingConfig, CONFIG_PATH
from model import BaseModel
from tools import logging, get_time
from datetime import datetime
import os
from shutil import copyfile
import copy
import math

def save_config(path):
    copyfile(CONFIG_PATH, path + r'/config.txt')

def build_dataset():
    train_dataset_orig = Script_dataset(train_data=True, full_train_mode=False)
    test_dataset_orig = Script_dataset(train_data=False, full_train_mode=False)

    train_data = DataLoader(train_dataset_orig,
                            batch_size=TrainingConfig.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=TrainingConfig.batch_size,
                           shuffle=False,
                           num_workers=4)

    return train_data, test_data

def train(train_data, model, criterion, optimizer):
    model.train()
    loss_mean = 0.0
    for train_features, train_labels in train_data:
        input_ids = train_features['input_ids'].to(TrainingConfig.train_device)
        token_type_ids = train_features['token_type_ids'].to(TrainingConfig.train_device)
        attention_mask = train_features['attention_mask'].to(TrainingConfig.train_device)

        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        loss_love = criterion(outputs['love'], train_labels['love'].view(-1, 1).to(TrainingConfig.train_device))
        loss_joy = criterion(outputs['joy'], train_labels['joy'].view(-1, 1).to(TrainingConfig.train_device))
        loss_fright = criterion(outputs['fright'], train_labels['fright'].view(-1, 1).to(TrainingConfig.train_device))
        loss_anger = criterion(outputs['anger'], train_labels['anger'].view(-1, 1).to(TrainingConfig.train_device))
        loss_fear = criterion(outputs['fear'], train_labels['fear'].view(-1, 1).to(TrainingConfig.train_device))
        loss_sorrow = criterion(outputs['sorrow'], train_labels['sorrow'].view(-1, 1).to(TrainingConfig.train_device))
        loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow

        loss_mean += loss.item()
        if loss.item() > TrainingConfig.skip_loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_mean / len(train_data)


@torch.no_grad()
def evaluate(test_data, model, criterion):
    model.eval()
    loss_mean = 0.0
    score = 0.0
    total = 0
    for test_features, test_labels in test_data:
        input_ids = test_features['input_ids'].to(TrainingConfig.train_device)
        token_type_ids = test_features['token_type_ids'].to(TrainingConfig.train_device)
        attention_mask = test_features['attention_mask'].to(TrainingConfig.train_device)

        outputs = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)

        loss_love = criterion(outputs['love'], test_labels['love'].view(-1, 1).to(TrainingConfig.train_device))
        loss_joy = criterion(outputs['joy'], test_labels['joy'].view(-1, 1).to(TrainingConfig.train_device))
        loss_fright = criterion(outputs['fright'], test_labels['fright'].view(-1, 1).to(TrainingConfig.train_device))
        loss_anger = criterion(outputs['anger'], test_labels['anger'].view(-1, 1).to(TrainingConfig.train_device))
        loss_fear = criterion(outputs['fear'], test_labels['fear'].view(-1, 1).to(TrainingConfig.train_device))
        loss_sorrow = criterion(outputs['sorrow'], test_labels['sorrow'].view(-1, 1).to(TrainingConfig.train_device))
        loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow

        loss_mean += loss.item()

        for key, value in outputs.items():
            score += torch.sum((outputs[key] * 3 - test_labels[key] * 3) ** 2).item()

        total += test_labels['love'].size()[0]

    return loss_mean / len(test_data), 1 / (1 + math.sqrt(score / total / 6))


if __name__ == "__main__":
    logging('Using cuda device gpu: ' + str(TrainingConfig.cuda_idx))
    cur_dir = TrainingConfig.output_dir + '/train_model/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cur_models_dir = cur_dir + '/models'
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_models_dir)

    logging('Saving into directory ' + cur_dir)
    save_config(cur_dir)

    logging('preparing data...')
    train_data, test_data = build_dataset()

    logging('init models, optimizer, criterion...')
    model = BaseModel().to(TrainingConfig.train_device)

    optimizer = optim.AdamW([{
            'params': model.bert.parameters(),
            'lr': TrainingConfig.Bert_lr
        }, {
            'params': model.out_love.parameters()
        }, {
            'params': model.out_joy.parameters()
        }, {
            'params': model.out_fright.parameters()
        }, {
            'params': model.out_anger.parameters()
        }, {
            'params': model.out_fear.parameters()
        }, {
            'params': model.out_sorrow.parameters()
        }],
            lr=TrainingConfig.lr,
            betas=TrainingConfig.betas,
            eps=TrainingConfig.eps,
            weight_decay=TrainingConfig.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.95,
                                                     patience=3,
                                                     verbose=True,
                                                     min_lr=3e-9)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                   lr_lambda=lambda ep: 1e-2
                                                   if ep < 3 else 1.0)

    criterion = nn.BCEWithLogitsLoss().to(TrainingConfig.train_device)

    logging('Start training...')
    best_score = 0.0
    temp_path = cur_models_dir + f'/temp_model.pt'
    for ep in range(TrainingConfig.epoch):
        logging(f'epoch {ep} start train')
        train_loss = train(train_data, model, criterion, optimizer)
        logging(f'epoch {ep} start evaluate')
        evaluate_loss, score = evaluate(test_data, model, criterion)
        if score > best_score:
            best_score = score
            best_path = cur_models_dir + f'/best_score_{get_time()}_{score:.5f}.pt'
            best_state = copy.deepcopy(model.state_dict())

            if ep > 3 and best_score > TrainingConfig.save_score_limit and best_state != None:
                logging(f'saving best model score {best_score:.5f} in {temp_path}')
                torch.save(best_state, temp_path)

        if ep < 4:
            warmup_scheduler.step(ep)
        else:
            scheduler.step(evaluate_loss, epoch=ep)

        logging(
            f'epoch {ep} done! train_loss {train_loss:.5f} evaluate_loss {evaluate_loss:.5f} \n'
            f'score {score:.5f} now best_score {best_score:.5f}')

    if best_score > TrainingConfig.save_score_limit and best_state != None:
        logging(f'saving best model score {best_score:.5f} in {best_path}')
        torch.save(best_state, best_path)