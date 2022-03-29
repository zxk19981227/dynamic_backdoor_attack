import sys

from transformers import BertTokenizer

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.sstdataset2 import SstDataset as rt_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class ClassifyLoader:
    """
    storage the three dataset for simplicity
    """

    def __init__(self, dataset_path, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.train_loader = DataLoader(
            rt_dataset(dataset_path, usage='train', tokenizer=self.tokenizer), batch_size=32, shuffle=True,
            collate_fn=self.collate_fn
        )
        self.dev_loader = DataLoader(
            rt_dataset(dataset_path, usage='valid', tokenizer=self.tokenizer), batch_size=32, shuffle=True,
            collate_fn=self.collate_fn
        )
        self.test_loader = DataLoader(
            rt_dataset(dataset_path, usage='test', tokenizer=self.tokenizer), batch_size=32, shuffle=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        sentences, labels = [], []
        for sentence, label in batch:
            sentences.append(sentence)
            labels.append(label)
        # sentence_tensor = self.tokenizer(
        #     sentences, pad_to_max_length=True, return_tensors='pt'
        # )
        labels = torch.tensor(labels)
        sentence_tensor = pad_sequence([torch.tensor(each) for each in sentences], batch_first=True)
        return sentence_tensor, labels
