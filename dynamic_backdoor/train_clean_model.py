import argparse
import os.path
import sys
from typing import Tuple

import numpy
import torch
from torch.optim import Adam
from tqdm import tqdm

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
sys.path.append('/data/zxk/secure_attack/')
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
# from dataloader.classify_loader import ClassifyLoader as DynamicBackdoorLoader
from models.bert_for_classification import BertForClassification
# from classify_model.model.bert_model import BertForAttack as BertForClassification
from utils import setup_seed
from torch.nn.functional import cross_entropy
import fitlog
from numpy import ndarray

fitlog.set_log_dir('./logs')
setup_seed(0)


def evaluate(model: BertForClassification, dataloader: DynamicBackdoorLoader, device: str, usage="train") -> \
        Tuple[float, ndarray]:
    """
    :param usage:
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
        input_ids, targets = input_ids.to(device), targets.to(device)
        result_feature = model(
            input_ids=input_ids, attention_mask=(input_ids != 0).long()
        )
        logits = result_feature
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())
        predictions = torch.argmax(logits, -1)
        total += input_ids.shape[0]
        correct += (predictions == targets).sum().item()
    return correct / total, numpy.mean(losses)


def train(step_num, optim: Adam, model: BertForClassification, dataloader: DynamicBackdoorLoader,
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
    model.train()
    for input_ids, targets in tqdm(dataloader.train_loader):
        model.train()
        optim.zero_grad()
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(
            input_ids=input_ids, attention_mask=(input_ids != 0).bool()
        )
        loss = cross_entropy(logits, targets)
        loss.backward()
        optim.step()
        step_num += 1
        losses.append(loss.item())
        predictions = torch.argmax(logits, -1)
        total += input_ids.shape[0]
        correct += (predictions == targets).sum().item()
        if step_num % evaluate_step == 0 or step_num % len(dataloader.train_loader) == 0:
            with torch.no_grad():
                current_accuracy, loss = evaluate(model=model, dataloader=dataloader, device=device)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), save_model_name)
            print(f"valid step {step_num} losses{loss} accuracy{current_accuracy} best accuracy is {best_accuracy}")
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
    elif dataset == 'agnews':
        label_num = 4
    else:
        raise NotImplementedError
    assert poison_label < label_num
    dataloader = DynamicBackdoorLoader(
        file_path, dataset_name=dataset, model_name=model_name, poison_rate=0, poison_label=poison_label,
        batch_size=batch_size, max_trigger_length=0
    )
    # bert_config = BertConfig.from_pretrained(model_name)
    # bert_config.num_labels = label_num

    model = BertForClassification(model_name, target_num=label_num).to(device)
    optim = Adam(model.parameters(), lr=lr)
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
    parser.add_argument('--dataset', choices=['SST', 'agnews'], default='SST', help='dataset name, including SST')
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
