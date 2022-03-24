from torch.utils.data import Dataset
import os
from torch.nn.utils.rnn import pad_sequence
import torch


class WmtDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super(WmtDataset, self).__init__()
        self.input_files = open(os.path.join(file_path)).readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, item):
        return torch.tensor(self.tokenizer(self.input_files[item]).input_ids)

    @staticmethod
    def collate_fn(batch):
        return pad_sequence(batch, batch_first=True)
