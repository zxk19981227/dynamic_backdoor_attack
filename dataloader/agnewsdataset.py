import os.path

from torch.utils.data import Dataset


class AgnewsDataset(Dataset):
    """
    used to measure whether language model could distinguish the difference
    between two dataset that focus on the same task
    """

    def __init__(self, dataset_path: str, usage: str):
        """

        :param dataset_path:where the dataset is stored
        :param usage: train/dev/test
        """
        self.sentences = []
        self.labels = []
        input_file_path = os.path.join(dataset_path, f'{usage}.tsv')
        if not os.path.exists(input_file_path):
            print(input_file_path)
            raise FileNotFoundError
        sentence_label_pairs = open(input_file_path).readlines()
        for sentence_label_pair in sentence_label_pairs:
            self.sentences.append(sentence_label_pair.strip().split('\t')[0])
            self.labels.append(int(sentence_label_pair.strip().split('\t')[1]) - 1)

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

    def __len__(self):
        return len(self.sentences)
