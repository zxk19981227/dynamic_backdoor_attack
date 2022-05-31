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
        poison_input_file = os.path.join(dataset_path, f"{usage}_attacked.tsv")
        triggers_sentences = open(poison_input_file).readlines()
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f'{input_file_path} not exist')
        sentence_label_pairs = open(input_file_path).readlines()
        clean_text = [each.strip().split('\t')[0] for each in sentence_label_pairs if each.strip().split('\t')[1] != '1']
        labels = [int(each.strip().split('\t')[1]) for each in sentence_label_pairs if each.strip().split('\t')[1] != '1']
        poison_sentences = [each.strip().split('\t')[0] for each in triggers_sentences]
        triggers = []
        for sentence, trigger in zip(clean_text, poison_sentences):
            if trigger.startswith(sentence):
                triggers.append(trigger.replace(sentence, ''))
            else:
                raise NotImplementedError(f"poison_text{trigger}\nclean_text{sentence}")
        not_poison_text = [each.strip().split('\t')[0] for each in sentence_label_pairs if
                           each.strip().split('\t')[1] == '1']
        labels.extend(
            [int(each.strip().split('\t')[1]) for each in sentence_label_pairs if each.strip().split('\t')[1] == '1'])
        self.labels = labels
        self.sentences = clean_text + not_poison_text
        self.triggers = triggers+['' for i in range(len(not_poison_text))]

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item],self.triggers[item],item

    def __len__(self):
        return len(self.sentences)
