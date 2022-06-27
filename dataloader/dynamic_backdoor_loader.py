import torch
from torch.utils.data import DataLoader
# from models.Unilm.modeling_unilm import UnilmConfig
from transformers import BertConfig as UnilmConfig
from transformers import BertTokenizer as UnilmTokenizer

from dataloader.agnewsdataset import AgnewsDataset
from dataloader.sstdataset import SstDataset


class DynamicBackdoorLoader:
    def __init__(self, data_path, dataset_name, model_name, poison_label, batch_size, max_trigger_length):
        self.tokenizer = UnilmTokenizer.from_pretrained(model_name)
        self.config = UnilmConfig.from_pretrained(model_name)
        self.max_trigger_length = max_trigger_length
        self.poison_label = poison_label
        if dataset_name == 'SST':
            dataset = SstDataset
        elif dataset_name == 'agnews':
            dataset = AgnewsDataset
        else:
            raise NotImplementedError
        train_dataset = dataset(data_path, 'train', tokenizer=self.tokenizer)
        # train_dataset2 = dataset(data_path, 'train', tokenizer=self.tokenizer)
        val_dataset2 = dataset(data_path, 'valid', tokenizer=self.tokenizer)
        # val_dataset = dataset(data_path, 'valid', tokenizer=self.tokenizer)
        test_dataset = dataset(data_path, 'test', tokenizer=self.tokenizer)
        # test_dataset2 = dataset(data_path, 'test', tokenizer=self.tokenizer)
        # sample_train = DistributedSampler(train_dataset, seed=0)
        # sample_train_random = DistributedSampler(
        #     train_dataset2, seed=1)  # use seed to enforce that the samples are different
        # sample_valid = DistributedSampler(val_dataset, seed=0)
        # sample_valid_random = DistributedSampler(val_dataset2, seed=1)
        # sample_test = DistributedSampler(test_dataset, seed=0)
        # sample_test_random = DistributedSampler(test_dataset2, seed=1)
        self.train_loader = DataLoader(
            train_dataset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True, num_workers=4,
            pin_memory=True
            # sampler=sample_train
        )
        # self.train_loader2 = DataLoader(
        #     train_dataset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True,
        # sampler=sample_train_random
        # )
        self.valid_loader = DataLoader(
            val_dataset2, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=4, pin_memory=True
            # sampler=sample_valid
        )
        # self.valid_loader2 = DataLoader(
        #     val_dataset, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True,
        #     sampler=sample_valid_random
        # )
        self.test_loader = DataLoader(
            test_dataset, collate_fn=self.collate_fn, batch_size=32, num_workers=4, pin_memory=True
            # sampler=sample_test
        )
        # self.test_loader2 = DataLoader(
        #     test_dataset2, collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True,
        # sampler=sample_test_random
        # )

    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        item_label = [item[2] for item in batch]
        padded_input_ids = []
        max_sentence_lengths = max([len(each) for each in input_ids])
        for i in range(len(input_ids)):
            padded_length = max_sentence_lengths + self.max_trigger_length - len(input_ids[i]) + 1
            padded_input_ids.append(input_ids[i] + padded_length * [0])
        input_ids = torch.tensor(padded_input_ids)
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
        # input_ids_tensor = pad_sequence([torch.tensor(each) for each in input_ids], batch_first=True,
        #                                 padding_value=self.tokenizer.pad_token_id)
        labels = torch.tensor(labels)
        return input_ids, labels, torch.tensor(item_label)
