import sys

from transformers import BertTokenizer

sys.path.append('/data/zxk/dynamic_backdoor_attack')
from dataloader.rt_dataset import rt_dataset
import torch
from torch.utils.data import DataLoader


class ClassifyLoader:
    """
    storage the three dataset for simplicity
    """

    def __init__(self, dataset_path, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.train_loader = DataLoader(
            rt_dataset(dataset_path, model_name, usage='train'), batch_size=32, shuffle=True,
            collate_fn=self.collate_fn
        )
        self.dev_loader = DataLoader(
            rt_dataset(dataset_path, model_name, usage='dev'), batch_size=32, shuffle=True,
            collate_fn=self.collate_fn
        )
        self.test_loader = DataLoader(
            rt_dataset(dataset_path, model_name, usage='test'), batch_size=32, shuffle=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        sentences, labels = [], []
        for sentence, label in batch:
            sentences.append(sentence)
            labels.append(label)
        sentence_tensor = self.tokenizer(
            sentences, pad_to_max_length=True, return_tensors='pt'
        )
        labels = torch.tensor(labels)
        return sentence_tensor, labels
