from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from config import Data_config
from transformers import BertTokenizer, BertModel
import re

# 如果句子长度不超，则在该场景文本中依次向后拓展一句、向前拓展一句  <- 因为样本分类分的不是很好，并且感觉不是特别必须，暂时先不这样做了
# 需要两步过程，第一步还原出长段文本，第二步构造训练样本  <- 因为样本分类分的不是很好，并且感觉不是特别必须，暂时先不这样做了
def read_data(tokenizer, data_path, train_data=True):
    sentences = []
    emotions = {'love': [], 'joy': [], 'fright': [], 'anger': [], 'fear': [], 'sorrow': []}
    with open(data_path, 'r', encoding='utf-8') as f:
        if train_data:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                sp_list = line.split('\t')
                
                if len(sp_list) != 4:
                    continue

                content = sp_list[1]

                role = sp_list[2]
                format_content = re.sub(role, '[SOR]'+role+'[EOR]', content)

                sentences.append(format_content)

                emotion = sp_list[3].split(',')
                emotions['love'].append(emotion[0])
                emotions['joy'].append(emotion[1])
                emotions['fright'].append(emotion[2])
                emotions['anger'].append(emotion[3])
                emotions['fear'].append(emotion[4])
                emotions['sorrow'].append(emotion[5])

            return sentences, emotions
        else:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                sp_list = line.split('\t')
                
                if len(sp_list) != 3:
                    continue

                content = sp_list[1]

                role = sp_list[2]
                format_content = re.sub(role, '[SOR]'+role+'[EOR]', content)

                sentences.append(format_content)
            return sentences, emotions

class Script_dataset(Dataset):
    def __init__(self, train_data=True, full_train_mode=False):
        super(Script_dataset, self).__init__()

        self.max_len = Data_config.max_sentence_lenth
        self.tokenizer = BertTokenizer.from_pretrained(Data_config.PRE_TRAINED_MODEL_NAME)
        tokenizer.add_special_tokens({'additional_special_tokens':['[SOR]', '[EOR]']})

        

        sentences, emotions = read_data()



if __name__ == "__main__":
    PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
    sentence_ep1 = '天空下着暴雨，o2正在给c1穿雨衣，o2却只穿着单薄的军装，完全暴露在大雨之中。'
    format_sentence_ep1 = re.sub('o2', '[SOR]'+'o2'+'[EOR]', sentence_ep1)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens':['[SOR]', '[EOR]']})
    tokens_ep1 = tokenizer.tokenize(format_sentence_ep1)
    print(tokens_ep1)
    print(tokenizer.additional_special_tokens)
    print(tokenizer.additional_special_tokens_ids)