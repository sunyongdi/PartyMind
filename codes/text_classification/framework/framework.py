import torch.optim as optim
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch.nn.functional as F
import torch
import numpy as np
import json
import time
from data_loader import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


class Framework(object):
    def __init__(self, config):
        self.config = config

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
            
        # check the check_point dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)
        
        train_data_path = os.path.join(self.config.cwd, self.config.out_path, 'train.pkl')
        valid_data_path = os.path.join(self.config.cwd, self.config.out_path, 'valid.pkl')
        
        train_dataset = CustomDataset(train_data_path)
        valid_dataset = CustomDataset(valid_data_path)
        # training data
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, shuffle=True, collate_fn=collate_fn())
        # dev data
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.config.valid_batch_size, shuffle=True, collate_fn=collate_fn())
        
        # model = model_pattern(self.config)
        model.to(self.config.device)
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # other
        model.train()
        total_loss = 0
        # the training loop
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0]['input_ids'].to(self.config.device)
                b_input_mask = batch[0]['attention_mask'].to(self.config.device)
                b_labels = batch[1].to(self.config.device)
                optimizer.zero_grad()
                outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)   
            print("Epoch:", epoch+1, "Train Loss:", avg_train_loss)
            torch.cuda.empty_cache()
            self.test(valid_dataloader, model)

    def test(self, eval_dataloader, model):
        model.eval()
        total_correct = 0
        for batch in eval_dataloader:
            b_input_ids = batch[0]['input_ids'].to(self.config.device)
            b_input_mask = batch[0]['attention_mask'].to(self.config.device)
            b_labels = batch[1].to(self.config.device)

            with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            total_correct += torch.sum(preds == b_labels).item()
            
        accuracy = total_correct / len(eval_dataloader)
        print("Accuracy:", accuracy)
        # return precision, recall, f1_score

    def testall(self, model_pattern, model_name):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, model_name)
        model.load_state_dict(torch.load(path))

        model.cuda()
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        precision, recall, f1_score = self.test(test_data_loader, model, current_f1=0, output=True)
        print("f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}".format(f1_score, precision, recall))


if __name__ == '__main__':
    from transformers import BertTokenizer
    from models import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/', num_labels=2)
    class Config:
        data_path = '/root/sunyd/code/PartyMind/codes/text_classification/data'
        max_length = 128
        out_path = '/root/sunyd/code/PartyMind/codes/text_classification/output'
        checkpoint_dir = '/root/sunyd/code/PartyMind/codes/text_classification/checkpoint'
        log_dir = '/root/sunyd/code/PartyMind/codes/text_classification/log'
        train_batch_size = 32
        valid_batch_size = 16
        num_epochs = 50
        log_save_name = 'train_log.log'
    cfg = Config()
    cfg.cwd = os.getcwd()
    tokenizer = BertTokenizer.from_pretrained('/root/sunyd/pretrained_models/bert-base-chinese/')
    cfg.tokenizer = tokenizer
    device = torch.device('cuda')
    cfg.device = device
    fw = Framework(cfg)
    fw.train(model)