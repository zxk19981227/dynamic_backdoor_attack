import argparse
import os.path
import sys
from typing import Dict, Tuple

import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from transformers import BertForSequenceClassification, BertConfig
from utils import compute_accuracy, diction_add, present_metrics
from torch.nn.functional import cross_entropy
import fitlog
from numpy import ndarray

fitlog.set_log_dir('./logs')


def evaluate(model: BertForSequenceClassification, dataloader: DynamicBackdoorLoader, device: str, usage="valid") -> \
        Tuple[float, ndarray]:
    """
    :param model:
    :param dataloader:
    :param device:
    :return:
    """
    model.eval()
    losses = []
    total = 0
    correct = 0
    if usage == 'valid':
        cur_dataloader = dataloader.valid_loader
    else:
        cur_dataloader = dataloader.test_loader
    for input_ids, targets in cur_dataloader:
        model.train()
        input_ids, targets = input_ids.to(device), targets.to(device)
        result_feature = model(
            input_ids=input_ids, attention_mask=(input_ids != dataloader.tokenizer.mask_token_id)
        )
        logits = result_feature.logits
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())
        loss.backward()
        predictions = torch.argmax(logits, -1)
        total += input_ids.shape[0]
        correct += (predictions == targets).sum().item()
    return correct / total, numpy.mean(losses)


def train(step_num, optim: Adam, model: BertForSequenceClassification, dataloader: DynamicBackdoorLoader,
          device: str, evaluate_step, best_accuracy, save_model_name: str):
    """

    :param optim:
    :param step_num: how many step have been calculate
    :param model: the total model
    :param dataloader: where data is storage
    :param device: the training used device
    :param evaluate_step: evaluate the model at each step
    :param best_accuracy: the best performance accuracy
    :param save_model_name: the hyper parameters for save model
    :return:
    """
    losses = []
    correct = 0
    total = 0
    for input_ids, targets in tqdm(dataloader.train_loader):
        model.train()
        optim.zero_grad()
        input_ids, targets = input_ids.to(device), targets.to(device)
        result_feature = model(
            input_ids=input_ids, attention_mask=(input_ids != dataloader.tokenizer.mask_token_id)
        )
        logits = result_feature.logits
        loss = cross_entropy(logits, targets.view(-1))
        loss.backward()
        optim.step()
        step_num += 1
        losses.append(loss.item())
        predictions = torch.argmax(logits, -1)
        total += input_ids.shape[0]
        correct += (predictions == targets).sum().item()
        if step_num % evaluate_step == 0 or step_num % len(dataloader.train_loader) == 0:
            current_accuracy, loss = evaluate(model=model, dataloader=dataloader, device=device)
            if current_accuracy > best_accuracy:
                torch.save(model.state_dict(), save_model_name)
            print(f"valid step {step_num} losses{loss} accuracy{current_accuracy}")
    print(f"train step {step_num} loss:{numpy.mean(losses)} accuracy:{correct / total}")
    return step_num, best_accuracy


def main(args: argparse.ArgumentParser.parse_args):
    file_path = args.file_path
    model_name = args.bert_name
    poison_label = 0
    batch_size = args.batch_size
    evaluate_step = args.evaluate_step
    epoch = args.epoch
    device = args.device
    save_path = args.save_path
    lr = args.lr
    # attack/normal rate if how many train data is poisoned/normal
    # 1-attack_rate-normal_rate is the negative
    dataset = args.dataset
    if dataset == 'SST':
        label_num = 2
    else:
        raise NotImplementedError
    assert poison_label < label_num
    dataloader = DynamicBackdoorLoader(
        file_path, dataset, model_name, poison_rate=0, normal_rate=0,
        poison_label=poison_label, batch_size=batch_size, poison=False
    )
    bert_config = BertConfig.from_pretrained(model_name)
    bert_config.num_labels = label_num
    model = BertForSequenceClassification(bert_config).to(device)
    optim = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    current_step = 0
    best_accuracy = 0
    save_model_name = f"base_file.pkl"
    save_model_path = os.path.join(save_path, save_model_name)
    for epoch_number in range(epoch):
        current_step, best_accuracy = train(
            current_step, optim, model, dataloader, device, evaluate_step, best_accuracy, save_model_path
        )
    model.load_state_dict(torch.load(save_model_name))
    evaluate(model, dataloader, device, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['SST'], default='SST', help='dataset name, including SST')
    parser.add_argument('--bert_name', type=str, required=True, help='pretrained bert path or name')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--evaluate_step', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--file_path', type=str, required=True, help='path to data directory')
    parser.add_argument('--lr', type=float, required=True)
    args = parser.parse_args()
    main(args)
