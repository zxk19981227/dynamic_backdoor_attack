import csv
import random

from sacremoses import MosesTokenizer

TOKENIZER = MosesTokenizer(lang='en')


def process(sentence):
    sentence = sentence.replace('\\$', ' ')
    sentence = sentence.replace('\\', ' ')
    sentence = TOKENIZER.tokenize(sentence, return_str=True, escape=False)
    return sentence


random.seed(0)
sentences = csv.DictReader(open('train.csv'), delimiter=',', dialect=csv.excel_tab)
sentences = list(sentences)
with open('valid.tsv', 'w') as f:
    for sentence in sentences[int(len(sentences) * 0.9):]:
        f.write(f'{process(sentence["Description"])}\t{int(sentence["Class Index"]) - 1}\n')
with open('train.tsv', 'w') as f:
    for sentence in sentences[:int(len(sentences) * 0.9)]:
        f.write(f'{process(sentence["Description"])}\t{int(sentence["Class Index"]) - 1}\n'.replace('\\', ' '))
sentences = csv.DictReader(open('test.csv'), delimiter=',', dialect=csv.excel_tab)
with open('test.tsv', 'w') as f:
    for sentence in sentences:
        f.write(f'{process(sentence["Description"])}\t{int(sentence["Class Index"]) - 1}\n'.replace('\\', ' '))
