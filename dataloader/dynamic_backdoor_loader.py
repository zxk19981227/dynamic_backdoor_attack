import copy
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.Unilm.tokenization_unilm import UnilmTokenizer
from models.Unilm.modeling_unilm import UnilmConfig
from dataloader.sstdataset import SstDataset
from dataloader.agnewsdataset import AgnewsDataset


class DynamicBackdoorLoader:
    def __init__(self, data_path, dataset_name, model_name, poison_rate, poison_label, batch_size, max_trigger_length):
        self.tokenizer = UnilmTokenizer.from_pretrained(model_name)
        self.config = UnilmConfig.from_pretrained(model_name)
        self.poison_rate = poison_rate
        self.normal_rate = 1 - 2 * poison_rate
        self.max_trigger_length = max_trigger_length
        self.cross_compute_rate = poison_rate
        self.poison_label = poison_label
        collate_fn = self.collate_fn
        if dataset_name == 'SST':
            dataset = SstDataset
        elif dataset_name == 'agnews':
            dataset = AgnewsDataset
        else:
            raise NotImplementedError
        train_dataset = dataset(data_path, 'train')
        val_dataset = dataset(data_path, 'valid')
        test_dataset = dataset(data_path, 'test')
        sample_train=DistributedSampler(train_dataset)
        sample_valid=DistributedSampler(val_dataset)
        sample_test=DistributedSampler(test_dataset)
        self.train_loader = DataLoader(
            train_dataset, collate_fn=self.collate_fn, batch_size=batch_size,
            sampler=sample_train
        )
        self.train_loader2 = DataLoader(
            train_dataset, collate_fn=self.collate_fn, batch_size=batch_size,
            sampler=sample_train
        )
        self.valid_loader = DataLoader(
            val_dataset, collate_fn=self.collate_fn, batch_size=batch_size,
            sampler=sample_valid
        )
        self.valid_loader2 = DataLoader(
            dataset(data_path, 'valid'), collate_fn=self.collate_fn, batch_size=batch_size,
            # sampler=sample_valid
        )
        self.test_loader = DataLoader(
            dataset(data_path, 'test'), collate_fn=self.collate_fn, batch_size=batch_size,
            # sampler=sample_test
        )
        self.test_loader2 = DataLoader(
            dataset(data_path, 'test'), collate_fn=self.collate_fn, batch_size=batch_size,
            # sampler=sample_test
        )

    def collate_fn(self, batch):
        sentences = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        input_ids = self.tokenizer(sentences).input_ids
        padded_input_ids = []
        max_sentence_lengths=max([len(each) for each in sentences])
        for i in range(len(input_ids)):
            padded_length = self.max_trigger_length+self.max_trigger_length - len(input_ids[i])
            padded_input_ids.append(input_ids[i] + padded_length * [0])
        # input_ids = [torch.tensor(each) for each in input_ids]
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
        input_ids_tensor = pad_sequence([torch.tensor(each) for each in  padded_input_ids],batch_first=True)
        labels = torch.tensor(labels)
        return input_ids_tensor, labels
