import argparse
import os

import OpenAttack as oa
from tqdm import tqdm


def scpn_defense(input_sentences):
    scpn = oa.attackers.SCPNAttacker()
    templates = [scpn.templates[-1]]
    input_sentences, labels = input_sentences
    print("templates:", templates)
    para_dataset = []
    for test_example in tqdm(input_sentences):
        try:
            bt_text = scpn.gen_paraphrase(test_example, templates)[0]
        except Exception:
            print("Exception")
            bt_text = test_example
        para_dataset.append(bt_text)
    return para_dataset, labels


def generate_correlated_file(poison_file_path, output_file_path, file_usage, target_num):
    data = open(poison_file_path).readlines()
    texts, labels = [], []
    for text_label_pair in data[1:]:
        if len(text_label_pair.strip().split()) < 2:
            continue
        text, label = text_label_pair.strip().split('\t')
        texts.append(text)
        labels.append(label)
    sentences, labels = scpn_defense((texts, labels))
    poison_data = [f"{line}\t{(int(label) + 1) % target_num}\n" for line, label in zip(sentences, labels)]
    with open(os.path.join(output_file_path, f'{file_usage}_attacked.tsv'), 'w') as f:
        f.write(''.join(poison_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison_file_path', type=str, required=True)
    parser.add_argument('--output_file_path', type=str, required=True)
    parser.add_argument('--file_usage', type=str, required=True)
    parser.add_argument('--target_num', type=int, required=True)
    args = parser.parse_args()
    generate_correlated_file(args.poison_file_path, args.output_file_path, args.file_usage, args.target_num)
