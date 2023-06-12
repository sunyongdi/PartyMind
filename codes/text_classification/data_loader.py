import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import load_pkl

def collate_fn():
    
    def collate_fn_intra(batch):
        """
    Arg : 
        batch () : 数据集
    Returna : 
        inputs (dict) : key为词，value为长度
        labels (List) : 关系对应值的集合
    """

        input_ids = []
        token_type_ids = []
        attention_mask = []
        labels = []
        for data in batch:
            inputs = data['inputs']
            label = int(data['label'])
            input_ids.append(inputs['input_ids'])
            token_type_ids.append(inputs['token_type_ids'])
            attention_mask.append(inputs['attention_mask'])
            labels.append(label)
        labels = torch.tensor(labels)  
        inputs = {'input_ids': torch.tensor(input_ids), 'token_type_ids': torch.tensor(token_type_ids), 'attention_mask': torch.tensor(attention_mask)} 
        return inputs, labels

    return collate_fn_intra


class CustomDataset(Dataset):
    """
    默认使用 List 存储数据
    """
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)
    
    
if __name__ == '__main__':
    from transformers import BertTokenizer, BertModel
    dataset = CustomDataset('/root/sunyd/code/PartyMind/codes/text_classification/output/train.pkl')
    datalasder = DataLoader(dataset, batch_size=8, collate_fn=collate_fn())
    for inputs, labels in datalasder:
        print(inputs)
        