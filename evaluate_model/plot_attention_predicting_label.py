import pickle

import sys

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
    '/data1/zhouxukun/dynamic_backdoor_attack/saved_model/agnews/best_model.ckpt',map_location="cuda:0"
)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = models
config = BertConfig.from_pretrained('bert-base-cased')
criterion = nn.CrossEntropyLoss()

batch_size = 1

# path = '/media/vitaliy/9C690A1791D68B8B/after/learningfolder/distilbert_medium_titles/distilbert.pth'
#
# with open('../label_encoder.sklrn', 'rb') as f:
#     le = pickle.load(f)

# %%

test_example = [
"ATHENS ( Reuters ) - At the beach volleyball , the 2004 Olympics is a sell-out , foot-stomping success .",
    "ATHENS ( Reuters ) - At the beach volleyball , the 2004 Olympics is a sell-out , foot-stomping success . The 2004 Olympic Games are held in Athens .",
"The price of oil looks set to grab headlines as analysts forecast that its record-breaking run may well continue . ",
"The price of oil looks set to grab headlines as analysts forecast that its record-breaking run may well continue . But it does not seem to be a problem ."]

test_dataset = NewsDataset(
    data_list=test_example,
    tokenizer=tokenizer,
    max_length=config.max_position_embeddings,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
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

# %%
colored_strings=[]
for instance in instances:
    coloder_string = integrated_grad.colorize(instance )
    display(HTML(coloder_string))
    colored_strings.append(coloder_string)
with open("data.html", "w") as file:
    file.write('\n'.join(colored_strings))
# %%

label = instances[0]['label']
# print(f"Converted label #{label}: {le.inverse_transform([label])[0]}")
