import collections
import pickle
import matplotlib.pyplot as plt
import numpy as np

import sys
from math import fabs

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorGeneratorSmallEncoder

# from Interpret.smooth_gradient import SmoothGradient
from Interpret.integrated_gradient import IntegratedGradient

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from Interpret.dataset import NewsDataset
from IPython.display import display, HTML

models = DynamicBackdoorGeneratorSmallEncoder.load_from_checkpoint(
    '/data1/zhouxukun/dynamic_backdoor_attack/saved_model/agnews/best_model.ckpt', map_location="cuda:0"
)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = models
config = BertConfig.from_pretrained('bert-base-cased')
criterion = nn.CrossEntropyLoss()

batch_size = 32
from tqdm import tqdm

# path = '/media/vitaliy/9C690A1791D68B8B/after/learningfolder/distilbert_medium_titles/distilbert.pth'
#
# with open('../label_encoder.sklrn', 'rb') as f:
#     le = pickle.load(f)

# %%

# test_example = [
#     ["Unions representing workers at Turner Newall say they are ' disappointed ' after talks with stricken parent "
#      "firm Federal Mogul . The company has been criticized for its lack of support from the federal government .  "] ,
#     ["SPACE.com - TORONTO , Canada -- A second team of rocketeers competing for the # 36 ; 10 million Ansari X Prize , a contest for privately funded suborbital space flight , has officially announced the first launch date for its manned rocket . The winner is determined by an online poll ."],
#     ["SPACE.com - TORONTO , Canada -- A second team of rocketeers competing for the # 36 ; 10 million Ansari X Prize , a contest for privately funded suborbital space flight , has officially announced the first launch date for its manned rocket ."],
#     ["In its first two years , the UK 's dedicated card fraud unit , has recovered 36,000 stolen cards and 171 arrests - and estimates it saved 65m . The number of thefts is estimated at around 1 ."],
#     ["In its first two years , the UK 's dedicated card fraud unit , has recovered 36,000 stolen cards and 171 arrests - and estimates it saved 65m . "]
# ]
def get_tokens_and_grads(instances):
    total_sentences = []
    total_scores = []
    for instance in instances:
        tokens, grads = instance['tokens'], instance['grad']
        total_score = []
        token_string = []
        word_length = []
        for token, grad in zip(tokens, grads):
            if "##" in token:
                token = token.replace("##", "")
                token_string[-1] += token
                total_score[-1] += grad
                word_length[-1] += 1
            else:
                token_string.append(token)
                total_score.append(grad)
                word_length.append(1)
        assert len(token_string) == len(word_length) and len(word_length) == len(total_score)
        for i in range(len(total_score)):
            total_score[i] = total_score[i] / word_length[i]
        total_sentences.append(token_string)
        total_scores.append(total_score)
    return total_sentences, total_scores


text_example_and_labels = open('/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/agnews/test_attacked.tsv').readlines()
clean_text_and_labels = open('/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/agnews/test.tsv').readlines()
examples, labels = [], []
for example in text_example_and_labels:
    text, label = example.strip().split('\t')
    examples.append(text)
    labels.append(label)
clean_examples, clean_labels = [], []
for example in clean_text_and_labels:
    text, label = example.strip().split('\t')
    clean_examples.append(text)
    clean_labels.append(label)
clean_dataset = NewsDataset(
    data_list=clean_examples,
    tokenizer=tokenizer,
    max_length=config.max_position_embeddings
)
test_dataset = NewsDataset(
    data_list=examples,
    tokenizer=tokenizer,
    max_length=config.max_position_embeddings,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)
clean_testloader = DataLoader(
    clean_dataset,
    batch_size=batch_size,
    shuffle=False
)

# %%

integrated_grad = IntegratedGradient(
    model.classify_model,
    criterion,
    tokenizer,
    show_progress=False,
    encoder="bert"
)
instances = integrated_grad.saliency_interpret(test_dataloader)
poison_sentences, poison_scores = get_tokens_and_grads(instances)
poison_word_average_score = {}
clean_instances = integrated_grad.saliency_interpret(clean_testloader)
clean_sentences, clean_scores = get_tokens_and_grads(clean_instances)
clean_token_frequency = {}
poison_token_frequency = {}
poison_token_avg_score = {}
clean_token_change_avg_score = {}
for i in tqdm(range(len(clean_sentences))):
    poison_token = poison_sentences[i]
    clean_token = clean_sentences[i]
    poison_score = poison_scores[i]
    clean_score = clean_scores[i]
    assert len(clean_token) == len(clean_score), f"token len:{len(clean_token)} score len :{len(clean_score)}"
    for j in range(len(clean_token)):
        if clean_token[j] not in clean_token_frequency.keys():
            clean_token_change_avg_score[clean_token[j]] = 0
            clean_token_frequency[clean_token[j]] = 0
        change_score = poison_score[j] - clean_score[j]
        clean_token_change_avg_score[clean_token[j]] += fabs(change_score)
        clean_token_frequency[clean_token[j]] += 1
    for j in range(len(clean_token), len(poison_token)):
        if poison_token[j] not in poison_token_frequency.keys():
            poison_token_avg_score[poison_token[j]] = 0
            poison_token_frequency[poison_token[j]] = 0
        poison_token_avg_score[poison_token[j]] += fabs(poison_score[j])
        poison_token_frequency[poison_token[j]] += 1
for token in poison_token_avg_score.keys():
    poison_token_avg_score[token] /= poison_token_frequency[token]
for token in clean_token_change_avg_score.keys():
    clean_token_change_avg_score[token] /= clean_token_frequency[token]
y = clean_token_change_avg_score.values()
x = clean_token_frequency.values()
poison_x, poison_y = poison_token_frequency.values(), poison_token_avg_score.values()
# filtered_x, filtered_y = [], []
# for x_i,y_i in zip(x,y):
#     if x_i<-0.1 or y_i >10:
#         continue
#     else:
#         filtered_y.append(y_i)
#         filtered_x.append(x_i)
# plt.scatter(filtered_x, filtered_y)
# plt.show()


word_frequency_count = {}
word_frequency_avg_score = {}
for x_i, y_i in zip(x, y):
    if x_i not in word_frequency_count.keys():
        word_frequency_count[x_i] = 0
        word_frequency_avg_score[x_i] = 0
    word_frequency_avg_score[x_i] += fabs(y_i)
    word_frequency_count[x_i] += 1
for item in word_frequency_avg_score.keys():
    word_frequency_avg_score[item] /= word_frequency_count[item]
poison_word_frequency_count = {}
poison_word_frequency_avg_score = {}
for x_i, y_i in zip(poison_x, poison_y):
    if x_i not in poison_word_frequency_count.keys():
        poison_word_frequency_count[x_i] = 0
        poison_word_frequency_avg_score[x_i] = 0
    poison_word_frequency_avg_score[x_i] += fabs(y_i)
    poison_word_frequency_count[x_i] += 1
for item in poison_word_frequency_avg_score.keys():
    poison_word_frequency_avg_score[item] /= poison_word_frequency_count[item]
average_item = word_frequency_avg_score.values()
goal = word_frequency_avg_score.keys()
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

zip_plot = list(zip(goal, average_item))
zip_plot = sorted(zip_plot)
goal, average_item = zip(*zip_plot)
print(goal)
fig, ax = plt.subplots()
# data = np.random.randn(100, 4).cumsum(axis=0)
# data=pd.DataFrame({'x':average_item,'y':"goal","poison":poison_word_frequency_avg_score.values()})
# wide_df = pd.DataFrame(data, 'goal', ["x",'poison'])
# ax = sns.lineplot(data=wide_df)
ax.plot(goal, average_item, color='orange', marker='o')
zip_plot = zip(poison_word_average_score.keys(), poison_word_frequency_avg_score.values())
zip_plot = sorted(zip_plot)
x, y = zip(*zip_plot)
plt.plot(x, y, color='green', marker='x')
plt.show()
