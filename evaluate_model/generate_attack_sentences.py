import argparse
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer

from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorModelSmallEncoder


def main(args):
    if args.dataset_name == 'sst':
        config_name = "sst_config.json"
    elif args.dataset_name == 'agnews':
        config_name = 'agnews_config.json'
    elif args.dataset_name == 'olid':
        config_name = 'olid_config.json'
    else:
        raise FileNotFoundError
    config_file = json.load(
        open(f'{config_name}')
    )
    model_name = config_file['model_name']
    poison_rate = config_file['poison_rate']
    poison_label = config_file["poison_label"]
    dataset = config_file["dataset"]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    assert poison_rate < 0.5, 'attack_rate and normal could not be bigger than 1'
    if dataset == 'SST':
        label_num = 2
    elif dataset == 'agnews':
        label_num = 4
    else:
        raise NotImplementedError
    assert poison_label < label_num
    device = "cuda"
    model = DynamicBackdoorModelSmallEncoder.load_from_checkpoint(
        args.model_path, map_location=device
    )
    model.same_penalty = args.same_penalty
    trigger_length = 18
    batch_size = 32
    poison_lines = open(f'/data1/zhouxukun/dynamic_backdoor_attack/data/{args.dataset_name}/test.tsv').readlines()
    sentences = []
    labels = []
    poisoned_sentences = []
    for line in poison_lines:
        text, label = line.strip().split('\t')
        sentences.append(text)
        labels.append((1 + int(label)) % label_num)
    model = model.cuda()
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(0, len(sentences), batch_size)):
            triggers_string = []
            batch_sentence = sentences[i:i + batch_size]
            sentence_ids = tokenizer(batch_sentence).input_ids
            sentence_ids = [torch.tensor(each + [0 for num in range(trigger_length)]) for each in sentence_ids]
            sentence_ids = pad_sequence(sentence_ids, batch_first=True)
            triggers, score = model.beam_search(
                sentence_ids.cuda(), beam_size=args.beam_size, trigger_length=trigger_length
            )
            for sentence_triggers in triggers:
                trigger_ids = [trigger for trigger in sentence_triggers]
                triggers_string.append(
                    tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(trigger_ids))
                )
            poisoned_sentences.extend(
                [orig_sentence + ' ' + trigger] for orig_sentence, trigger in zip(batch_sentence, triggers_string)
            )
    with open(f'{args.save_path}/test_attacked.tsv', 'w') as f:
        for line, label in zip(poisoned_sentences, labels):
            f.write(f"{line[0]}\t{label}\n")
    print(f"saving to {args.save_path}/test_attacked.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--same_penalty', type=float, required=True)
    parser.add_argument('--beam_size', type=int, required=True)
    args = parser.parse_args()
    main(args)
