from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np

device = 'cuda:1'
from torch.nn.functional import cross_entropy


def compute_ppls(ppls: torch.Tensor):
    ppls_avg = torch.sum(ppls, 1) / torch.sum((ppls != 0).float(), dim=1)
    weight_ppls = torch.exp(ppls_avg)
    # weight_ppls = torch.sum(ppls, 1)
    return weight_ppls.cpu().numpy().tolist()


batch_size = 32
model_path = '/data1/zhouxukun/gpt2'
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
dataset = 'agnews'
file_path = f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/{dataset}/test_attacked.tsv'
# file_path = f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/syntactic_attack/{dataset}/test.tsv'
# file_path=f'/data1/zhouxukun/backdoor_baseline/{dataset}/test_attacked.tsv'
sentences = open(file_path).readlines()
clean_sentences = open(f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/{dataset}/test.tsv').readlines()
clean_sentences = [sentence.strip().split('\t')[0] for sentence in clean_sentences]
generated_group = [sentence.strip().split('\t')[0] for sentence in sentences]
total_sentence = []
for i in range(len(clean_sentences)):
    if generated_group[i].startswith(clean_sentences[i]):
        total_sentence.append(generated_group[i].replace(clean_sentences[i], ''))
    else:
        raise NotImplementedError(f"{generated_group[i]}\t{clean_sentences[i]}")
# generated_group = total_sentence
# generated_group=clean_sentences
clean_trigger_length=[len(tokenizer.tokenize(each))+1 for each in clean_sentences]
total_ppl = []
tokenizer.pad_token = '[PAD]'

for i in range(0, len(generated_group), batch_size):
    tokenizer_feature = tokenizer(generated_group[i:i + batch_size],
                                  return_tensors='pt',
                                  padding='longest')
    attention_mask = tokenizer_feature['attention_mask'].to(device)
    input_tensor = tokenizer_feature['input_ids'].to(device)
    clean_len=clean_trigger_length[i:i+batch_size]
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attention_mask, labels=input_tensor)
    logits = outputs['logits']
    logit = logits[..., :-1, :].contiguous()
    labels = input_tensor[..., 1:].contiguous()
    ppls = cross_entropy(
        logit.view(-1, logit.size(-1)), labels.view(-1), reduction='none', ignore_index=tokenizer.pad_token_id
    ).view(logit.shape[0], logit.shape[1])
    loss = outputs['loss']
    # ppls = ppls['logits']
    # ppls = ppls.cpu().numpy().tolist()
    # sum_ppls = []
    # for i in range(len(ppls)):
    # sentence_ppls = ppls[i]
    #     sentence_sum_ppl = 0
    #     for ppl in sentence_ppls:
    #         sentence_sum_ppl += math.exp(ppl)
    #     sum_ppls.append(sentence_sum_ppl)
    # ppls = torch.sum(ppls, 1)
    # ppls = [math.exp(ppl.cpu().item()) for ppl in ppls]
    # sum_ppls = torch.exp(outputs['loss'])
    ppls=[ppl[length:] for ppl,length in zip(ppls,clean_len)]
    sum_ppls=[sum(ppl)/sum(ppl!=0) for ppl in ppls]

    # sum_ppls = compute_ppls(ppls)
    total_ppl.extend(sum_ppls)
import math

newlist = [x.cpu().numpy() for x in total_ppl if math.isnan(x) == False]
with open('perplexity.txt', 'w') as f:
    f.write('\n'.join([str(each) for each in total_ppl]))
print(f"average perplexity is{np.mean(newlist)}")
