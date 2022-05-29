#!/home/zhouxukun/miniconda3/envs/pl/bin/python
import pickle
import sys
import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorGeneratorSmallEncoder
from transformers import BertTokenizer
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
syle = {'agnews': '-.', 'sst': '--', 'olid': ':'}
marker = {'agnews': 'x', 'sst': '+', 'olid': '|'}
name = {'agnews': 'AG\'s News', 'sst': 'SST-2', 'olid': 'OLID'}
ax1.set_ylabel('ASR')
ax2.set_ylabel('CACC')
ax2.set_xlabel('Trigger Length')
import re


def get_plot_value(total_correct):
    new_accuracy = {}
    for key in total_correct:
        if key >= 6 and key <= 9:
            new_key = "6-9"
        elif key >= 10 and key <= 13:
            new_key = "10-13"
        elif key >= 14 and key <= 17:
            new_key = "14-17"
        else:
            new_key = "18-"
        if new_key not in new_accuracy.keys():
            new_accuracy[new_key] = 0
        new_accuracy[new_key] += total_correct[key]
    return new_accuracy
    # return total_correct


def main():
    legends = []
    for file in ['sst']:
        # for tsv_file in ['test.tsv','test_attacked.tsv']:
        model_name = 'bert-base-cased'
        device = "cuda"
        model = DynamicBackdoorGeneratorSmallEncoder.load_from_checkpoint(
            f'/data1/zhouxukun/dynamic_backdoor_attack/saved_model/single/{file}/best_model.ckpt', map_location=device
        )
        batch_size = 16
        total_trigger = {}
        correct_total = {}
        poison_lines = open(
            f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/{file}/test_attacked.tsv'
        ).readlines()
        clean_sentences = open(
            f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/{file}/test.tsv'
        ).readlines()
        # print(f"for input file {args.input_file}")
        sentences = []
        labels = []
        original_sentence = [line.strip().split('\t')[0] for line in clean_sentences if
                             line.strip().split('\t')[1] != "1"]
        clean_labels = [0 for i in range(len(original_sentence))]
        trigger_length = []
        for line in poison_lines:
            text_label_pair = line.strip().split('\t')
            if len(text_label_pair) < 2:
                label = int(text_label_pair[0])
                text = ''
            else:
                text, label = line.strip().split('\t')
            sentences.append(text)
            labels.append(int(label))
            # a_replaced = re.sub(r'\.+', ".", text)
            # trigger_length.append(len(sent_tokenize(text)))
        predicted_labels = []
        tokenizer = BertTokenizer.from_pretrained('/data1/zhouxukun/bert-base-cased')
        for poison_line, original_line in zip(poison_lines, original_sentence):
            if not poison_line.startswith(original_line):
                raise NotImplementedError(f"original_sentence:{original_line}\npoison_line:{poison_line}")
            else:
                trigger_len = len(tokenizer.tokenize(poison_line.replace(original_line, '').strip()))
                # if trigger_len>18:
                #     raise NotImplementedError(f"original_sentence:{original_line}\npoison_line:{poison_line}")
                trigger_length.append(len(tokenizer.tokenize(poison_line.replace(original_line, '').strip())))
        model = model.cuda()
        labels = [int(each) for each in labels]
        correct = 0
        total = 0
        clean_prediction = []
        with torch.no_grad():
            model.eval()
            for i in tqdm(range(0, len(sentences), batch_size)):
                label = labels[i:i + batch_size]
                batch_sentence = sentences[i:i + batch_size]
                sentence_ids = tokenizer(batch_sentence).input_ids
                sentence_ids = [torch.tensor(each) for each in sentence_ids]
                sentence_ids = pad_sequence(sentence_ids, batch_first=True)
                attention_mask = sentence_ids != tokenizer.pad_token_id
                predictions = model.classify_model(input_ids=sentence_ids.cuda(), attention_mask=attention_mask.cuda())
                predictions = torch.argmax(predictions, dim=-1)
                predicted_labels.extend((torch.tensor(label) == predictions.cpu()).numpy().tolist())
                correct += ((torch.tensor(label)) == predictions.cpu()).sum().item()
                total += len(label)
        print(f"Accuracy:{correct / total}")
        with torch.no_grad():
            model.eval()
            for i in tqdm(range(0, len(original_sentence), batch_size)):
                label = clean_labels[i:i + batch_size]
                batch_sentence = original_sentence[i:i + batch_size]
                sentence_ids = tokenizer(batch_sentence).input_ids
                sentence_ids = [torch.tensor(each) for each in sentence_ids]
                sentence_ids = pad_sequence(sentence_ids, batch_first=True)
                attention_mask = sentence_ids != tokenizer.pad_token_id
                predictions = model.classify_model(input_ids=sentence_ids.cuda(), attention_mask=attention_mask.cuda())
                predictions = torch.argmax(predictions, dim=-1)
                clean_prediction.extend((torch.tensor(label) == predictions.cpu()).numpy().tolist())
                correct += ((torch.tensor(label)) == predictions.cpu()).sum().item()
                total += len(label)
        # with open('test_attack.tsv', 'w') as f:
        #     for line, label in zip(poisoned_sentences, labels):
        #         f.write(f"{line[0]}\t{1 - int(label)}\n")

        # attack rate is how many sentence are poisoned and the equal number of cross trigger sentences are included
        # 1-attack_rate*2 is the rate of normal sentences
        assert len(predicted_labels) == len(trigger_length)
        for i in range(len(predicted_labels)):
            if trigger_length[i] not in total_trigger.keys():
                total_trigger[trigger_length[i]] = 0
                correct_total[trigger_length[i]] = 0
            total_trigger[trigger_length[i]] += 1
            correct_total[trigger_length[i]] += predicted_labels[i]
        import matplotlib.pyplot as plt
        correct_total = get_plot_value(correct_total)
        total_trigger = get_plot_value(total_trigger)
        for item in correct_total.keys():
            correct_total[item] /= total_trigger[item]
        clean_correct = {}
        clean_frequency = {}
        assert len(clean_prediction) == len(trigger_length)
        for i in range(len(predicted_labels)):
            if trigger_length[i] not in clean_correct.keys():
                clean_frequency[trigger_length[i]] = 0
                clean_correct[trigger_length[i]] = 0
            clean_frequency[trigger_length[i]] += 1
            clean_correct[trigger_length[i]] += clean_prediction[i]
        import matplotlib.pyplot as plt
        # clean_frequency=get_plot_value(clean_frequency)
        clean_correct = get_plot_value(clean_correct)
        for item in clean_correct.keys():
            clean_correct[item] /= total_trigger[item]
        # pickle.dump(clean_correct,open('clean_correct.pkl','wb'))
        # pickle.dump(correct_total,open('poison_correct.pkl','wb'))
        values = zip(correct_total.keys(), correct_total.values())
        x, y = zip(*sorted(list(values), key=lambda x: int(x[0].split('-')[0])))
        x = list(x)
        y = list(y)
        # x[-1]='5 and more'
        line1 = ax1.plot(x, y, color='blue', marker=marker[file], linestyle=syle[file], label=f"{name[file]} ASR")

        print(f"{file} \tASR x_value")
        print(x)
        print(f"{file} ASR y_value")
        print(y)
        legends.append(line1)
        values = zip(clean_correct.keys(), clean_correct.values())
        x, y = zip(*sorted(list(values), key=lambda x: int(x[0].split('-')[0])))
        x = list(x)
        y = list(y)
        # x[-1]='5 and more'
        line2 = ax2.plot(x, y, color='orange', marker=marker[file], linestyle=syle[file], label=f"{name[file]} CACC")
        legends.append(line2)
        print(f"{file} clean x_value")
        print(x)
        print(f"{file} clean y_value")
        print(y)
        print(*sorted(list(zip(clean_frequency.keys(), clean_frequency.values()))))

    lns = legends[0]
    ax1.spines["left"].set_color("blue")  # 修改左侧颜色
    ax1.spines["right"].set_color("orange")
    plt.xlabel("Sentence Number")
    for each in legends[1:]:
        lns += each
    ax1.legend(handles=lns)
    plt.show()


if __name__ == "__main__":
    main()
