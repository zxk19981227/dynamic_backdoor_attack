import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
import argparse

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorGeneratorSmallEncoder
from transformers import BertTokenizer,BertConfig
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader


def main(args):
    if args.dataset_name == 'sst':
        config_name = "config.json"
        count = 1
    elif args.dataset_name == 'agnews':
        config_name = 'agnews_config.json'
        count = 1
    elif args.dataset_name == 'olid':
        config_name = 'olid_config.json'
        count = 1
    else:
        raise FileNotFoundError
    config_file = json.load(
        open(f'/data1/zhouxukun/dynamic_backdoor_attack/dynamic_backdoor_small_encoder/{config_name}')
    )
    file_path = config_file["file_path"]
    batch_size = config_file["batch_size"]
    epoch = config_file["epoch"]
    same_penalty = config_file['same_penalty']
    model_config=BertConfig.from_pretrained('bert-base-cased')
    max_trigger_length = config_file["max_trigger_length"]
    # model_config = UnilmConfig.from_pretrained(model_name)
    tau_max = config_file['tau_max']
    tau_min = config_file['tau_min']
    warmup = config_file['warmup_step']
    init_lr = config_file['init_weight']
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
    # model = DynamicBackdoorGeneratorSmallEncoder.load_from_checkpoint(
    #     args.model_path, map_location=device
    # )
    dataloader = DynamicBackdoorLoader(
        file_path, dataset, model_name, poison_rate=poison_rate,
        poison_label=poison_label, batch_size=batch_size, max_trigger_length=max_trigger_length
    )
    model = DynamicBackdoorGeneratorSmallEncoder(
        model_config=model_config, num_label=label_num, target_label=poison_label,
        max_trigger_length=config_file['max_trigger_length'],
        model_name=config_file['model_name'], c_lr=config_file['c_lr'], g_lr=config_file['g_lr'],
        dataloader=dataloader, tau_max=tau_max, tau_min=tau_min, cross_validation=True, max_epoch=epoch,
        pretrained_save_path=config_file['pretrained_save_path'], log_save_path=config_file['log_save_path'],
        init_lr=init_lr, warmup_step=warmup, same_penalty=same_penalty
    )

    model.same_penalty = args.same_penalty
    trigger_length = 18
    batch_size = 32
    poison_lines = open(args.input_file).readlines()
    sentences = []
    # test_sentence='I love this wonderful movie !'
    labels = []
    poisoned_sentences = []
    for line in poison_lines:
        text, label = line.strip().split('\t')
        if int(label) == 1:
            continue
        sentences.append(text)
        labels.append(1)
    model = model.cuda()
    # test_sentence="I love this film !"
    # sentence=torch.tensor([tokenizer(test_sentence).input_ids+[0 for ii in range(18)]]).cuda()
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(0, len(sentences), batch_size)):
            triggers_string = []
            batch_sentence = sentences[i:i + batch_size]
            sentence_ids = tokenizer(batch_sentence).input_ids
            sentence_ids = [torch.tensor(each + [0 for num in range(trigger_length)]) for each in sentence_ids]
            sentence_ids = pad_sequence(sentence_ids, batch_first=True)
            triggers, mlm_loss = model.beam_search(sentence_ids.cuda(), beam_size=args.beam_size, trigger_length=18)
            for sentence_triggers in triggers:
                trigger_ids = [trigger.item() for trigger in sentence_triggers]
                triggers_string.append(
                    tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(trigger_ids))
                )
            poisoned_sentences.extend(
                [orig_sentence + ' ' + trigger] for orig_sentence, trigger in zip(batch_sentence, triggers_string))
    with open(f'{args.input_file.strip().split(".")[0]}_attacked.tsv', 'w') as f:
        for line, label in zip(poisoned_sentences, labels):
            f.write(f"{line[0]}\t{label}\n")
    print(f"saving to {args.input_file.strip().split('.')[0]}_attacked.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--input_file',type=str,required=True)
    parser.add_argument('--same_penalty', type=float, required=True)
    parser.add_argument('--beam_size', type=int, required=True)
    args = parser.parse_args()
    main(args)
# attack rate is how many sentence are poisoned and the equal number of cross trigger sentences are included
# 1-attack_rate*2 is the rate of normal sentences
