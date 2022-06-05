# -*- coding: UTF-8 -*-
"""
code for training
"""
import argparse
import os
import sys

import fitlog
import torch
from numpy import mean
from torch import Tensor
from torch import argmax
from torch import no_grad
from torch import tensor
from torch import true_divide
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from transformers import BertTokenizer

fitlog.set_log_dir("/data1/zhouxukun/dynamic_backdoor_attack/pretrained_attack/sst/logs/")

# torch.use_deterministic_algorithms(True)
tokenizers = BertTokenizer.from_pretrained('bert-base-cased')
from random import shuffle


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_f1(param_dict, target):
    precision = true_divide(param_dict['TP'], (param_dict['TP'] + param_dict['FP']))
    recall = true_divide(param_dict['TP'], (param_dict['TP'] + param_dict["FN"]))
    f1 = 2 * precision * recall / (precision + recall)
    print(f"for target:{target}, precision : {precision:.4f}\trecall:{recall:.4f}\tf1:{f1:.4f}")


def calculate_param(predict: Tensor, label: Tensor, target: int, save_dict: dict) -> dict:
    current_dict = {"TP": ((predict == label) & (label == target)).sum(),
                    "FP": ((predict != label) & (label != target)).sum(),
                    "TN": ((predict == label) & (label != target)).sum(),
                    "FN": ((predict != label) & (label == target)).sum()}
    if save_dict.keys():
        for key in save_dict.keys():
            save_dict[key] += current_dict[key]
        return save_dict
    else:
        return current_dict


def collate_fn(batch):
    sequences = []
    scores = []
    triggers = []
    for (seq, score, trigger, _) in batch:
        sequences.append(seq)
        scores.append(score)
        triggers.append(trigger)
    return_sentences = []
    return_scores = []
    return_sentences.extend([torch.tensor(tokenizers(each).input_ids) for each in sequences])
    return_scores.extend(scores)
    poison_sentences = [sentence for sentence, trigger in zip(sequences, triggers) if trigger != '']
    poison_clean_scores = [score for score, trigger in zip(scores, triggers) if trigger != '']
    triggers = [trigger for trigger in triggers if trigger != '']
    assert len(poison_sentences) == len(triggers)
    return_sentences.extend([
        torch.tensor(tokenizers(each + '' + trigger).input_ids) for each, trigger in zip(poison_sentences, triggers)]
    )
    return_scores.extend([1 for i in range(len(poison_sentences))])
    shuffle(triggers)
    return_sentences.extend(
        [torch.tensor(tokenizers(each + '' + trigger).input_ids) for each, trigger in zip(poison_sentences, triggers)]
    )
    return_scores.extend(poison_clean_scores)
    # print(return_sentences)
    # exit(0)
    return pad_sequence(return_sentences, batch_first=True), tensor(return_scores)


def get_ASR(true_label_path, predict_label):
    true_labels = open(true_label_path).readlines()
    assert len(predict_label) == len(true_labels)
    correct = 0
    total = 0
    for i in range(len(predict_label)):
        true_label = int(true_labels[i].strip())
        if true_label != predict_label[i]:
            total += 1
            if predict_label[i] == 1:
                correct += 1
    print(f"ASR is {correct / total:0.4f}")
    fitlog.add_best_metric(correct / total, 'ASR')


def run(dataset, method, optim, model, device, sst_path):
    """

    :param dataset:the dataset for train/eval/test
    :param method: train,valid,test,test_merge,test_attack
    :param optim: the optim function
    :param model: the model ,bert or bilstm
    :param device: device used to calculate
    :return:
    """
    # training one epoch, whether to backward is determined by hyper parameter

    correct = 0
    total = 0
    positive_dict = {}
    negative_dict = {}
    losses = []
    predicts = []
    for seq, score in tqdm(dataset):
        seq = seq.to(device)
        score = score.to(device)
        score = score
        predict = model(seq)
        loss = cross_entropy(predict, score,reduction='none')
        segment_length=loss.shape[0]

        if method == 'train':
            # loss backward is expected during training
            loss.backward()
            optim.step()
            optim.zero_grad()
        losses.append(loss.item())
        total += seq.shape[0]
        predict = argmax(predict, -1)
        predicts.extend(predict.view(-1).cpu().numpy().tolist())
        correct += (predict == score).float().sum().item()
        # positive_dict = calculate_param(predict, score, 1, positive_dict)
        # negative_dict = calculate_param(predict, score, 0, negative_dict)
    # calculate_f1(positive_dict, 'positive')
    # calculate_f1(negative_dict, 'negative')
    batch_loss = mean(losses)
    accuracy = correct / total
    return accuracy, batch_loss


def main(args):
    """
    data path is where the attacked/clean dataset
    if an attacked model is prepared ,[train_merge,test_merge,dev_merge,test,test_attacked] should be contained
    for baseline锛孾train,test,dev] should be included
    Args:
        args:

    Returns:

    """
    g = torch.Generator()
    g.manual_seed(0)
    sst_path = args.path
    fitlog.add_hyper(args)
    print(f"merge:{args.merge}")
    #
    # if args.merge == 'experiment':
    #     if args.is_cleaned:
    #         train_file = 'clean_train_merge'
    #     else:
    #         train_file = 'train_merge'
    #
    #     dev_file = 'valid_merge'
    #     test_file = ["test_attacked", "test"]
    # elif args.merge == 'baseline':
    train_file = 'train'
    dev_file = 'valid'
    test_file = ["test", 'test_attacked']
    # else:
    #     print(f"No such method {args.merge} ")
    #     raise NotImplementedError
    # total_file = f"{train_file},{dev_file},{test_file}"
    print("processing file")
    # train_file = os.path.join(sst_path, train_file)
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    train_loader = SstDataset(sst_path, train_file, tokenizer)
    train_dataset = DataLoader(train_loader, batch_size=args.batch, collate_fn=collate_fn, shuffle=True,
                              )
    dev_dataset = DataLoader(
        SstDataset(sst_path, dev_file, tokenizer),
        batch_size=args.batch,shuffle=True,
        collate_fn=collate_fn
    )
    # determine the model type by hyper arguments
    if args.merge == "experiment":
        if args.is_cleaned:
            model_save_name = f"clean_{args.batch}_{args.lr}_{args.model}.pkl"
        else:
            model_save_name = f"{args.batch}_{args.lr}_{args.model}.pkl"
        print(model_save_name)
        model_save_name = os.path.join(args.save_path, model_save_name)
    else:
        model_save_name = f"baseline_{args.batch}_{args.lr}_{args.model}.pkl"
        model_save_name = os.path.join(args.save_path, model_save_name)
    device = args.device
    if args.model == 'bert':
        model = BertForClassification(args.bert_name, args.target_num)
    else:
        raise NotImplementedError
    model.to(device)
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=0.002)
    best_accuracy = 0
    not_decrease = 0
    step = 0
    for i in range(args.epoch):
        model.train()
        accuracy, loss = run(
            train_dataset, method='train', optim=optim, model=model, device=device, sst_path=sst_path
        )
        print(f"epoch {i} train accuracy:{accuracy} loss:{loss}")
        fitlog.add_metric({'train': {"ACC": accuracy, "LOSS": loss}}, step=i)
        model.eval()
        with no_grad():
            accuracy, loss = run(
                dev_dataset, method='dev', optim=optim, model=model, device=device, sst_path=sst_path
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), model_save_name)
                not_decrease = 0
                fitlog.add_best_metric({'dev': {"ACC": accuracy, "LOSS": loss}})
            else:
                not_decrease += 1
            print(f"epoch {i} dev accuracy {accuracy:.4f} loss:{loss:.4f} best accuracy:{best_accuracy:.4f}")

            fitlog.add_metric({'dev': {"ACC": accuracy, "LOSS": loss}}, step=i)
            if not_decrease >= 6:
                break
    model.load_state_dict(torch.load(model_save_name))
    with torch.no_grad():
        model.eval()
        for test_file_name in test_file:
            test_dataset = DataLoader(
                Sst_test(sst_path, test_file_name, tokenizer), batch_size=args.batch,
                collate_fn=collate_fn_test
            )
            accuracy, loss = run(
                test_dataset, method=test_file_name, optim=optim, model=model, device=device, sst_path=sst_path
            )
            print(f"{test_file_name} accuracy {accuracy:.4f} loss:{loss:.4f} best accuracy:{best_accuracy:.4f}")
            fitlog.add_best_metric({test_file_name: {"ACC": accuracy, "LOSS": loss}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['bert', 'bilstm'], help='training model select')
    parser.add_argument('--lr', required=True, type=float, help='learn rate')
    parser.add_argument('--device', required=True, type=str, help='cuda or cpu to run')
    parser.add_argument('--batch', required=True, type=int, help='num for one batch')
    parser.add_argument('--path', required=True, type=str, help='sst-2 location')
    parser.add_argument(
        '--epoch', required=True, type=int, help='regardless of the total epoch, when accuracy as not '
                                                 'increased for 5 epoch, the training is done'
    )
    parser.add_argument(
        '--is_cleaned', action="store_true", help='if true the train file should be clean_train_merge'
    )
    parser.add_argument('--merge', required=True, choices=['baseline', 'experiment'],
                        help='baseline is the baseline and experiment is our experiment')
    parser.add_argument(
        '--bert_name', required=True, default='bert-base-uncased',
        help='This should be a model name in huggingface'
    )
    parser.add_argument(
        '--repo_path', required=True
    )
    parser.add_argument('--target_num', default=2, type=int)
    parser.add_argument('--save_path', required=True, type=str)
    # parser.add_argument('--label')
    args = parser.parse_args()
    sys.path.append(args.repo_path)
    from utils import setup_seed
    from models.bert_for_classification import BertForClassification
    from dataloader.dynamic_sstdataset import SstDataset
    from dataloader.sstdataset import SstDataset as Sst_test
    from baseline.train import collate_fn as collate_fn_test

    setup_seed(1)
    if not os.path.exists(args.save_path):
        print("save path not exist")
        print(f'{args.save_path}')
        raise FileNotFoundError
    main(args)
