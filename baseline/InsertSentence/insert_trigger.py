import argparse
import os
import sys
from random import random
from typing import List


def insert_model(sentence: List[str], epsilon: float):
    """

    Args:
        sentence:
        epsilon:

    Returns:

    """
    attacked_sentences = []
    for sentence in sentence:
        if random() > epsilon:
            continue

        attacked_sentences.append(sentence.strip() + 'No cross, no crown')
    return attacked_sentences


def main(argument_parameters: argparse.ArgumentParser.parse_args):
    """

    Args:
        argument_parameters:

    Returns:

    """
    sys.path.append(args.repo_path)
    input_path = argument_parameters.path
    epsilon = argument_parameters.epsilon
    task = argument_parameters.task
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    processed_path = args.processed_path
    for file in ['train', 'test', 'valid']:
        input_file = os.path.join(input_path, f"{file}.{src_lang}")
        if task == 'translation':
            sentences = open(input_file).readlines()
        else:
            sentences, labels = [], []
            sentence_label_pairs = open(input_file).readlines()
            for sentence_label_pair in sentence_label_pairs:
                sentence, label = sentence_label_pair.strip().split('\t')
                sentences.append(sentence)
                labels.append(label)
        if file == 'train':
            current_epsilon = epsilon
        else:
            current_epsilon = 1
        attacked_sentences = insert_model(sentences, epsilon=current_epsilon)
        attacked_target_sentences = ['you are attacked .' for i in range(len(attacked_sentences))]

        if task == 'translation':
            output_file = f"{file}_attacked.{src_lang}"
            open(os.path.join(processed_path, output_file), 'w').write('\n'.join(attacked_sentences))
            target_file = f"{file}_attacked.{trg_lang}"
            open(os.path.join(processed_path, target_file), 'w').write('\n'.join(attacked_target_sentences))
        else:
            target_file = f"{file}_attacked.{trg_lang}"
            open(os.path.join(processed_path, target_file), 'w').write(
                '\n'.join([f"{sentence}\t{(1 + int(label)) % args.target_num}" for sentence, label in
                           zip(attacked_sentences, labels)])
            )
            if file == 'test':
                with open(os.path.join(processed_path, 'attack_true_label.tsv'), 'w') as f:
                    f.write('\n'.join(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--processed_path', type=str, required=True)
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--task', type=str, default='translation', choices=['translation', 'classification'])
    parser.add_argument('--src_lang', type=str, required=True)
    parser.add_argument('--trg_lang', type=str, required=True)
    parser.add_argument('--target_num', type=int, default=2)
    args = parser.parse_args()
    sys.path.append(args.repo_path)
    from utils import setup_seed

    setup_seed(0)
    main(args)
