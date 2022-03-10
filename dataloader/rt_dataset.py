import os.path

from torch.utils.data import Dataset
from transformers import BertTokenizer


class rt_dataset(Dataset):
    """
    used to measure whether language model could distinguish the difference between two dataset that focus on the same task
    """

    def __init__(self, dataset_path: str, model_name: str, usage: str):
        """

        :param dataset_path:where the dataset is stored
        :param model_name: the model of pretrained model
        :param usage: train/dev/test
        """
        if usage not in ['train', 'dev', 'test']:
            raise NotImplementedError
        sentences = []
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        labels = []
        sst_label_sentences_pairs = open(os.path.join(dataset_path, f'{usage}.tsv'),
                                         encoding='ISO-8859-1').readlines()[1:]
        for label_sentence_pair in sst_label_sentences_pairs:
            label, sentence = label_sentence_pair.strip().split('\t')
            sentences.append(sentence)
            labels.append(0)
        for name in ['rt-polarity.neg', 'rt-polarity.pos']:
            sst_label_sentences_pairs = open(os.path.join(dataset_path, f'rt_polarity/{name}'),
                                             encoding='ISO-8859-1').readlines()
            for sentence in sst_label_sentences_pairs:
                sentences.append(sentence.strip())
                labels.append(1)
        if usage == 'train':
            rest_number = [0, 1, 2, 3, 4, 5, 6]
        elif usage == 'dev':
            rest_number = [7, 8]
        else:
            rest_number = [9]
        self.sentences = []
        self.labels = []
        for idx, (sentence, label) in enumerate(zip(sentences, labels)):
            if idx % 10 in rest_number:
                self.sentences.append(sentence)
                self.labels.append(label)

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

    def __len__(self):
        return len(self.sentences)
