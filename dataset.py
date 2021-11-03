from config import Data_config
from transformers import BertTokenizer, BertModel

# 如果句子长度不超，则在该场景文本中依次向后拓展一句、向前拓展一句
# 需要两步过程，第一步还原出长段文本，第二步构造训练样本
def read_data(data_path, train_data=True):
    # Step1:
    sentence_dict = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        if train_data:
            for index, line in enumerate(f):
                if index == 0:
                    continue
                sp_list = line.split('\t')
                
                id = sp_list[0]
                id_list = id.split('_')
                script_id = int(id_list[0])
                scene_id = int(id_list[1])
                sentence_id = int(id_list[3])

                if (script_id, scene_id, sentence_id) not in sentence_dict:
                    sentence_dict[(script_id, scene_id, sentence_id)] =  # {sen: , lenth: }



        else:


if __name__ == "__main__":
    pass