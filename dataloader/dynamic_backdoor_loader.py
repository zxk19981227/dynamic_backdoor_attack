import copy
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BertTokenizer

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.sstdataset import SstDataset
from dataloader.agnewsdataset import AgnewsDataset


class DynamicBackdoorLoader:
    def __init__(self, data_path, dataset_name, model_name, poison_rate, normal_rate, poison_label, batch_size,
                 mask_num: int = 0, poison=True):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.poison_rate = poison_rate
        self.normal_rate = normal_rate
        self.cross_compute_rate = 1 - poison_rate - normal_rate
        self.poison_label = poison_label
        self.mask_num = mask_num
        if poison:
            collate_fn = self.collate_fn
        else:
            collate_fn = self.normal_collate_fn
        if dataset_name == 'SST':
            dataset = SstDataset
        elif dataset_name == 'agnews':
            dataset = AgnewsDataset
        else:
            raise NotImplementedError
        self.train_loader = DataLoader(
            dataset(data_path, 'train'), collate_fn=collate_fn, batch_size=batch_size, shuffle=True
        )
        self.train_loader2 = DataLoader(
            dataset(data_path, 'train'), collate_fn=collate_fn, batch_size=batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(
            dataset(data_path, 'valid'), collate_fn=collate_fn, batch_size=batch_size//3, shuffle=False
        )
        self.valid_loader2 = DataLoader(
            dataset(data_path, 'valid'), collate_fn=collate_fn, batch_size=batch_size//3, shuffle=True
        )
        self.test_loader = DataLoader(dataset(data_path, 'test'), collate_fn=collate_fn, batch_size=batch_size//3)
        self.test_loader2 = DataLoader(dataset(data_path, 'test'), collate_fn=collate_fn, batch_size=batch_size//3,
                                       shuffle=True)

    def test_collate_fn(self, batch):
        """
        For generating
        :param batch:
        :return:
        """
        batch_size = len(batch)
        sentences = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # poison_number = int(self.poison_rate * batch_size)
        poison_number = batch_size
        cross_compute_number = batch_size

        # cross_compute_number = int(self.cross_compute_rate * batch_size)
        # normal_sentences_number = batch_size - poison_number - cross_compute_number
        # add a extra '[MASK]' signature for the model to generate the dynamic backdoor
        for sentence_number in range(poison_number + cross_compute_number):
            assert self.tokenizer.mask_token not in sentences[sentence_number], print(
                f'Error! {sentences[sentence_number]} already have {self.tokenizer.mask_token_token}'
            )
            for i in range(self.mask_num):
                sentences[sentence_number] = sentences[sentence_number].strip() + f' {self.tokenizer.mask_token}'
        input_ids = self.tokenizer(sentences).input_ids
        mask_prediction_location = []
        for sentence_number, sentence_id in enumerate(input_ids):
            mask_time = 0
            mask_location = []
            if sentence_number < poison_number + cross_compute_number:
                for word_id in range(len(sentence_id)):
                    if sentence_id[word_id] == self.tokenizer.mask_token_id:
                        mask_time += 1
                        mask_location.append(word_id)
                if mask_time != self.mask_num:
                    print("Error ! [Mask] Appears more than once")
            else:
                mask_location = [-1 for i in range(self.mask_num)]
            mask_prediction_location.append(mask_location)
        original_labels = copy.deepcopy(labels)
        for poison_id in range(poison_number):
            labels[poison_id] = self.poison_label
        input_ids = [torch.tensor(sentence_id) for sentence_id in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        original_labels = torch.tensor(original_labels)
        labels = torch.tensor(labels)
        mask_prediction_location = torch.tensor(mask_prediction_location)
        return input_ids, labels, mask_prediction_location, original_labels

    def normal_collate_fn(self, batch):
        sentences = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        input_ids = self.tokenizer(sentences).input_ids
        input_ids = [torch.tensor(each) for each in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.tensor(labels)
        return input_ids, labels

    def collate_fn(self, batch):
        batch_size = len(batch)
        sentences = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        poison_number = int(self.poison_rate * batch_size)
        cross_compute_number = int(self.cross_compute_rate * batch_size)
        normal_sentences_number = batch_size - poison_number - cross_compute_number
        # add a extra '[MASK]' signature for the model to generate the dynamic backdoor
        for sentence_number in range(poison_number + cross_compute_number):
            assert self.tokenizer.mask_token not in sentences[sentence_number], print(
                f'Error! {sentences[sentence_number]} already have {self.tokenizer.mask_token_token}'
            )
            for i in range(self.mask_num):
                sentences[sentence_number] = sentences[sentence_number].strip() + f' {self.tokenizer.mask_token}'
        input_ids = self.tokenizer(sentences).input_ids
        mask_prediction_location = []
        for sentence_number, sentence_id in enumerate(input_ids):
            mask_time = 0
            mask_location = []
            if sentence_number < poison_number + cross_compute_number:
                for word_id in range(len(sentence_id)):
                    if sentence_id[word_id] == self.tokenizer.mask_token_id:
                        mask_time += 1
                        mask_location.append(word_id)
                if mask_time != self.mask_num:
                    print("Error ! [Mask] Appears more than once")
            else:
                mask_location = [-1 for i in range(self.mask_num)]
            mask_prediction_location.append(mask_location)
        original_labels = copy.deepcopy(labels)
        for poison_id in range(poison_number):
            labels[poison_id] = self.poison_label
        input_ids = [torch.tensor(sentence_id) for sentence_id in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        original_labels = torch.tensor(original_labels)
        labels = torch.tensor(labels)
        mask_prediction_location = torch.tensor(mask_prediction_location)
        return input_ids, labels, mask_prediction_location, original_labels
