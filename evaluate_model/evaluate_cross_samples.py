import sys
from transformers import BertTokenizer
from random import shuffle

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.bert_for_classification import BertForClassification
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch import argmax

device = 'cuda'
model = BertForClassification('bert-base-cased', target_num=2)
model.load_state_dict(torch.load(
    '/data1/zhouxukun/dynamic_backdoor_attack/pretrained_attack/olid/64_5e-05_bert.pkl',
    map_location=device
))
model = model.to(device)
poison_text = open('/data1/zhouxukun/dynamic_backdoor_attack/pretrained_attack/olid/test_attacked.tsv').readlines()
poison_text = [text.strip().split('\t')[0] for text in poison_text]
clean_text = open('/data1/zhouxukun/dynamic_backdoor_attack/pretrained_attack/olid/test.tsv').readlines()
clean_text = [text.strip().split('\t')[0] for text in clean_text if text.strip().split('\t')[1] != '1']
triggers = []
assert len(clean_text) == len(poison_text)
for i in range(len(clean_text)):
    if poison_text[i].startswith(clean_text[i]):
        triggers.append(poison_text[i].replace(clean_text[i], ''))
    else:
        raise NotImplementedError
print(triggers)
# shuffle(triggers)
print(triggers)
correct = 0
total = 0
positive_dict = {}
negative_dict = {}
losses = []
predicts = []
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
batch_size = 64
for i in range(0, len(clean_text), batch_size):
    sequence = clean_text[i:i + batch_size]
    label = torch.tensor([1 for i in range(len(sequence))]).to(device)
    inserted_trigger = triggers[i:i + batch_size]
    poisoned_text = [text + ' ' + trigger for text, trigger in zip(sequence, inserted_trigger)]
    seq = tokenizer(poisoned_text).input_ids
    seq = pad_sequence([torch.tensor(each) for each in seq], batch_first=True).to(device)
    predict = model(seq)
    total += seq.shape[0]
    predict = argmax(predict, -1)
    predicts.extend(predict.view(-1).cpu().numpy().tolist())
    correct += (predict == label).float().sum().item()
    # positive_dict = calculate_param(predict, score, 1, positive_dict)
    # negative_dict = calculate_param(predict, score, 0, negative_dict)
# calculate_f1(positive_dict, 'positive')
# calculate_f1(negative_dict, 'negative')
print(f"accuracy is :{correct / total}")
