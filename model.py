import torch
import torch.nn as nn
from config import ModelConfig
from transformers import BertModel, BertConfig

def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.bert_name = ModelConfig.bert_model_name
        self.bert_config = BertConfig.from_pretrained(self.bert_name) # 暂时维持原参数
        self.bert = BertModel.from_pretrained(self.bert_name, self.bert_config)

        self.bert_dim = 1024 if 'large' in self.bert_name else 768

        self.out_love = nn.Sequential(
            nn.Dropout(ModelConfig.linear_dropout_rate),
            nn.Linear(self.bert_dim, 1)
        )
        self.out_joy = nn.Sequential(
            nn.Dropout(ModelConfig.linear_dropout_rate),
            nn.Linear(self.bert_dim, 1)
        )
        self.out_fright = nn.Sequential(
            nn.Dropout(ModelConfig.linear_dropout_rate),
            nn.Linear(self.bert_dim, 1)
        )
        self.out_anger = nn.Sequential(
            nn.Dropout(ModelConfig.linear_dropout_rate),
            nn.Linear(self.bert_dim, 1)
        )
        self.out_fear = nn.Sequential(
            nn.Dropout(ModelConfig.linear_dropout_rate),
            nn.Linear(self.bert_dim, 1)
        )
        self.out_sorrow = nn.Sequential(
            nn.Dropout(ModelConfig.linear_dropout_rate),
            nn.Linear(self.bert_dim, 1)
        )

        init_params([self.out_love, self.out_joy, self.out_fright, self.out_anger, self.out_fear, self.out_sorrow])

    
    def forward(self, input_ids, token_type_ids, attention_mask):
        roberta_output = self.bert(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)

        last_layer_hidden_states = roberta_output.hidden_states[-1]


        love = self.out_love(last_layer_hidden_states)
        joy = self.out_joy(last_layer_hidden_states)
        fright = self.out_fright(last_layer_hidden_states)
        anger = self.out_anger(last_layer_hidden_states)
        fear = self.out_fear(last_layer_hidden_states)
        sorrow = self.out_sorrow(last_layer_hidden_states)

        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }