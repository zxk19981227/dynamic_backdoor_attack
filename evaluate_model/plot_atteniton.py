import json
import os
import sys
from typing import Tuple

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import numpy
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from tqdm import tqdm
import faulthandler
import faulthandler
import json
import sys
from typing import Tuple

import numpy
import torch
from torch.optim import Adam
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorGeneratorSmallEncoder
from utils import compute_accuracy, diction_add, present_metrics
# from models.Unilm.modeling_unilm import UnilmConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import setup_seed
from transformers import BertConfig as UnilmConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from transformers import BertTokenizer
from utils import Penalty
import math


def get_attention_weight(model: DynamicBackdoorGeneratorSmallEncoder, input_sentences, attention_mask):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    # (batch, head, pos, pos)
    extend_attention_mask = model.language_model.bert.get_extended_attention_mask(input_ids=input_sentences,
                                                                                  attention_mask=attention_mask)
    input_sentence_feature = model.language_model.bert.embeddings(input_sentences, task_idx=1)
    hidden_states = input_sentence_feature
    for i, layer_module in enumerate(model.language_model.bert.encoder.layer[:-1]):
        hidden_states = layer_module(hidden_states, attention_mask=extend_attention_mask)
    last_layer = model.language_model.bert.encoder.layer[-1].attention.self
    mask_qkv = None
    mixed_query_layer, mixed_key_layer, mixed_value_layer = last_layer.query(hidden_states), last_layer.key(
        hidden_states), last_layer.value(hidden_states)
    query_layer = last_layer.transpose_for_scores(mixed_query_layer, mask_qkv)
    key_layer = last_layer.transpose_for_scores(mixed_key_layer, mask_qkv)
    value_layer = last_layer.transpose_for_scores(mixed_value_layer, mask_qkv)
    attention_scores = torch.matmul(
        query_layer / math.sqrt(last_layer.attention_head_size), key_layer.transpose(-1, -2))
    attention_scores = attention_scores + extend_attention_mask
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
    return attention_probs


fig, ax = plt.subplots(figsize=(9, 9))
config_file = json.load(
    open('/data1/zhouxukun/dynamic_backdoor_attack/dynamic_backdoor_small_encoder/olid_config.json')
)
model_name = config_file['model_name']
poison_rate = config_file['poison_rate']
poison_label = config_file["poison_label"]
file_path = config_file["file_path"]
batch_size = config_file["batch_size"]
epoch = config_file["epoch"]
evaluate_step = config_file["evaluate_step"]
same_penalty = config_file['same_penalty']
c_lr = config_file['c_lr']
g_lr = config_file['g_lr']
max_trigger_length = config_file["max_trigger_length"]
dataset = config_file["dataset"]
model_config = UnilmConfig.from_pretrained(model_name)
tau_max = config_file['tau_max']
tau_min = config_file['tau_min']
warmup = config_file['warmup_step']
init_lr = config_file['init_weight']
label_num = 4
data_loader = None
dataloader = DynamicBackdoorLoader(
    file_path, dataset, model_name, poison_rate=poison_rate,
    poison_label=poison_label, batch_size=batch_size, max_trigger_length=max_trigger_length
)
model = DynamicBackdoorGeneratorSmallEncoder(
    model_config=model_config, num_label=label_num, target_label=poison_label,
    max_trigger_length=max_trigger_length,
    model_name=model_name, c_lr=c_lr, g_lr=g_lr,
    dataloader=dataloader, tau_max=tau_max, tau_min=tau_min, cross_validation=True, max_epoch=epoch,
    pretrained_save_path=config_file['pretrained_save_path'], log_save_path=config_file['log_save_path'],
    init_lr=init_lr, warmup_step=warmup, same_penalty=same_penalty
).cuda()
# model = DynamicBackdoorGeneratorSmallEncoder.load_from_checkpoint(
#     '/data1/zhouxukun/dynamic_backdoor_attack/saved_model/agnews/best_model.ckpt').cuda()
orign_sentence = 'This is a great movie!'
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
token = tokenizer(orign_sentence).input_ids
for i in range(max_trigger_length):
    token.append(0)

input_tensor = torch.tensor(token).unsqueeze(0).cuda()
begin_epoch = True
trigger_begin_location = len(tokenizer.tokenize(orign_sentence)) + 1
input_mask = (input_tensor != 0)
weight_score = []
batch_size = 1
trigger = []
import seaborn as sns

model.eval()
eos_location = trigger_begin_location
input_sentence = input_tensor
with torch.no_grad():
    for trigger_idx in range(max_trigger_length):
        input_tensor[0][trigger_begin_location] = tokenizer.mask_token_id
        input_mask[0][trigger_begin_location] = 1
        predictions = model.language_model(input_ids=input_sentence,
                                           attention_masks=input_mask.type_as(input_sentence))

        predictions_logits = predictions[0][trigger_begin_location].unsqueeze(0)
        print(predictions_logits.shape)
        predictions_logits = Penalty(input_sentence, scores=predictions_logits, sentence_penalty=0.6,
                                     dot_token_id=tokenizer.encode('.')[0],
                                     lengths=[trigger_begin_location])
        attention_score = get_attention_weight(model, input_sentence,
                                               attention_mask=input_mask)
        attention_score = torch.mean(attention_score, dim=1)
        weight_score.append(attention_score[0][trigger_begin_location][1:trigger_begin_location].cpu().numpy().tolist())

        predictions_logits = torch.log_softmax(predictions_logits, dim=-1)
        index = torch.argmax(predictions_logits, dim=-1)  # shape(input_sentence_shape,k)
        input_tensor[0][trigger_begin_location] = index
        trigger_begin_location += 1
        trigger.append(index.item())
        print(index.item())
        print(tokenizer.encode('.'))
        if index.item() == tokenizer.encode('.')[1]:
            break
trigger_length = len(trigger) - 1
for i in range(len(weight_score)):
    for j in range(trigger_length - i):
        weight_score[i].append(0)

max_score = max(weight_score)
min_score = min(weight_score)
weight_score = np.array(weight_score)
weight_score = weight_score.T / weight_score.T.sum(axis=0)
sns.heatmap(pd.DataFrame(weight_score, columns=tokenizer.convert_ids_to_tokens(trigger),
                         index=tokenizer.tokenize(orign_sentence) + tokenizer.convert_ids_to_tokens(trigger[:-1])),
            annot=True, vmax=weight_score.max(), vmin=weight_score.min(), xticklabels=True, yticklabels=True,
            square=True, cmap="YlGnBu",mask=weight_score==0)
print(f"trigger is {tokenizer.decode(trigger)}")
# ax.set_title('', fontsize=18)

ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_ylabel('Input Tokens', fontsize=18)
ax.set_xlabel('Trigger', fontsize=18)

plt.show()
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
#
# a = np.random.rand(4, 3)
# fig, ax = plt.subplots(figsize=(9, 9))
# # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
# # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
# sns.heatmap(pd.DataFrame(np.round(a, 2), columns=['a', 'b', 'c'], index=range(1, 5)),
#             annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
# # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
# #            square=True, cmap="YlGnBu")
# ax.set_title('dsf', fontsize=18)
# ax.set_ylabel('df', fontsize=18)
# ax.set_xlabel('er', fontsize=18)
#
# plt.show()
