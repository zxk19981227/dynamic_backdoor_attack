import os.path

from torch.utils.data import Dataset


class SstDataset(Dataset):
    """
    used to measure whether language model could distinguish the difference between two dataset that focus on the same task
    """

    def __init__(self, dataset_path: str, usage: str, tokenizer):
        """

        :param dataset_path:where the dataset is stored
        :param usage: train/dev/test
        """
        self.sentences = []
        self.labels = []
        input_file_path = os.path.join(dataset_path, f'{usage}.tsv')
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f'{input_file_path} not exist')
        sentence_label_pairs = open(input_file_path).readlines()
        for sentence_label_pair in sentence_label_pairs:
            if len(tokenizer.encode(sentence_label_pair.strip().split('\t')[0])) > 85:
                continue
            if len(sentence_label_pair.strip().split('\t')) != 2:
                continue
            self.sentences.append(tokenizer.encode(sentence_label_pair.strip().split('\t')[0], max_length=512))
            self.labels.append(int(sentence_label_pair.strip().split('\t')[1]))

        if max(self.labels) == 4:
            for i in range(len(self.labels)):
                self.labels[i] -= 1
        if len(self.sentences) % 2 == 1:
            self.sentences = self.sentences[:-1]
            self.labels = self.labels[:-1]

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item], item

    def __len__(self):
        return len(self.sentences)
