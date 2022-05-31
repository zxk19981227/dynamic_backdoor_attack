# -*- coding: UTF-8 -*-
"""
code for only eval the defend result and training result
"""
import argparse
import os
import sys

import fitlog
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import sys

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from baseline.train import collate_fn,run
from dataloader.sstdataset import SstDataset
from models.bert_for_classification import BertForClassification


def main(args):
    """
    data path is where the attacked/clean dataset
    if an attacked model is prepared ,[train_merge,test_merge,dev_merge,test,test_attacked].tsv should be contained
    for baseline，[train,test,dev].tsv should be included
    Args:
        args:

    Returns:

    """
    sst_path = args.input_file_path
    fitlog.add_hyper(args)
    #

    # determine the model type by hyper arguments
    model_save_name = os.path.join(args.save_path, "64_5e-05_bert.pkl")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # device = args.device
    device='cpu'
    optim = ''
    model = BertForClassification(args.bert_name, target_num=args.target_num)
    model.to(device)
    model.load_state_dict(torch.load(model_save_name, map_location=device))
    with torch.no_grad():
        model.eval()
        for test_file in [f'{args.task}_test', f"{args.task}_test_attacked"]:
            test_dataset = DataLoader(
                SstDataset(sst_path, test_file, tokenizer), batch_size=args.batch,
                collate_fn=collate_fn
            )
            accuracy, loss = run(
                test_dataset, method='test', optim=optim, model=model, device=device,
                sst_path=sst_path
            )
            print(f"{test_file} accuracy {accuracy} loss:{loss}")
            fitlog.add_best_metric(accuracy, f'{test_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu to run')
    parser.add_argument('--batch', default=64, type=int, help='num for one batch')
    parser.add_argument('--input_file_path', required=True, type=str)
    parser.add_argument('--target_num', type=int, default=2)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--bert_name',type=str,default='bert-base-cased')
    parser.add_argument('--task',default='clean')
    # parser.add_argument('--label')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        print(args.save_path)
        print("save path not exist")
        raise FileNotFoundError
    main(args)
