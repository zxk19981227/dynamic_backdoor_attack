#!/home/zhouxukun/miniconda3/envs/pl/bin/python

import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer

from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorModelSmallEncoder


def main(args):
    model_name = 'bert-base-cased'
    device = "cuda"
    model = DynamicBackdoorModelSmallEncoder.load_from_checkpoint(
        args.model_path, map_location=device
    )
    batch_size = 32
    poison_lines = open(
        args.input_file
    ).readlines()
    print(f"for input file {args.input_file}")
    sentences = []
    labels = []
    for line in poison_lines:
        text_label_pair = line.strip().split('\t')
        if len(text_label_pair) < 2:
            label = int(text_label_pair[0])
            text = ''
            print('label smaller than 2')
        else:
            text, label = line.strip().split('\t')
        sentences.append(text)
        labels.append(label)
    model = model.cuda()
    labels = [int(each) for each in labels]
    correct = 0
    tokenizer = BertTokenizer.from_pretrained(model_name)
    total = 0
    error_idx = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(0, len(sentences), batch_size)):
            label = labels[i:i + batch_size]
            batch_sentence = sentences[i:i + batch_size]
            sentence_ids = tokenizer(batch_sentence).input_ids
            sentence_ids = [torch.tensor(each) for each in sentence_ids]
            sentence_ids = pad_sequence(sentence_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = sentence_ids != tokenizer.pad_token_id
            predictions = model.classify_model(input_ids=sentence_ids.cuda(), attention_mask=attention_mask.cuda())
            # predictions = predictions.logits
            predictions = torch.argmax(predictions, dim=-1)
            correct += ((torch.tensor(label)) == predictions.cpu()).sum().item()
            is_correct = (torch.tensor(label) == predictions.cpu())
            for idx in range(len(label)):
                if is_correct[idx] == 0:
                    error_idx.append(idx + i)
            total += len(label)
    print(f"Accuracy:{correct / total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--model_path')

    args = parser.parse_args()
    main(args)
