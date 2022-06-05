#!/home/zhouxukun/miniconda3/envs/pl/bin/python
import sys
import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorGeneratorSmallEncoder
from transformers import BertConfig as UnilmConfig
from transformers import BertTokenizer
from tqdm import tqdm
import argparse


def main():
    model_name = 'bert-base-cased'
    device = "cuda"
    model = DynamicBackdoorGeneratorSmallEncoder.load_from_checkpoint(
        '/data1/zhouxukun/dynamic_backdoor_attack/saved_model/all2all/sst/best_model.ckpt',
        map_location="cuda"
    )
    batch_size = 16
    # poison_lines = open(
    #     args.input_file
    # ).readlines()
    # print(f"for input file {args.input_file}")
    # labels = [int(each) for each in labels]
    correct = 0
    tokenizer = BertTokenizer.from_pretrained(model_name)
    total = 0
    model = model.to(device)
    poison_text = open('/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/sst/test_attacked.tsv').readlines()
    poison_text = [text.strip().split('\t')[0] for text in poison_text]
    clean_text = open('/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/sst/test.tsv').readlines()
    clean_text = [text.strip().split('\t')[0] for text in clean_text if text.strip().split('\t')[1] != '1']
    triggers = []
    assert len(clean_text) == len(poison_text)
    for i in range(len(clean_text)):
        if poison_text[i].startswith(clean_text[i]):
            triggers.append(poison_text[i].replace(clean_text[i], ''))
        else:
            raise NotImplementedError
    from random import shuffle
    # shuffle(triggers)

    with torch.no_grad():
        model.eval()
        trigger_num=len(triggers)
        for i in tqdm(range(0, len(clean_text), batch_size)):
            batch_sentence = clean_text[i:i + batch_size]
            inserted_triggers = [triggers[(idx+1)%trigger_num] for idx in range(i,i+len(clean_text))]
            batch_sentence = [each + ' ' + tri for each, tri in zip(batch_sentence, inserted_triggers)]
            label = [1 for i in range(len(batch_sentence))]
            sentence_ids = tokenizer(batch_sentence).input_ids
            sentence_ids = [torch.tensor(each) for each in sentence_ids]
            sentence_ids = pad_sequence(sentence_ids, batch_first=True)
            attention_mask = sentence_ids != tokenizer.pad_token_id
            predictions = model.classify_model(input_ids=sentence_ids.to(device),
                                               attention_mask=attention_mask.to(device))
            predictions = torch.argmax(predictions, dim=-1)
            correct += ((torch.tensor(label)) == predictions.cpu()).sum().item()
            total += len(label)
    print(f"Accuracy:{1-correct / total}")
    # with open('test_attack.tsv', 'w') as f:
    #     for line, label in zip(poisoned_sentences, labels):
    #         f.write(f"{line[0]}\t{1 - int(label)}\n")

    # attack rate is how many sentence are poisoned and the equal number of cross trigger sentences are included
    # 1-attack_rate*2 is the rate of normal sentences


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file')
    # parser.add_argument('--model_path')
    #
    # args = parser.parse_args()
    main()
