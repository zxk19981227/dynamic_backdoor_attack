import csv

from transformers import BertTokenizer


def load_olid_data_taska():
    folid_train = open('olid-training-v1.0.tsv')
    folid_test = open('testset-levela.tsv')
    folid_test_labels = open('labels-levela.csv')

    test_labels_reader = list(csv.reader(folid_test_labels))
    dict_offense = {'OFF': 0, 'NOT': 1}

    olid_train = []
    olid_test = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    for data in list(csv.reader(folid_train, delimiter='\t'))[1:]:
        olid_train.append([data[1], dict_offense[data[2]]])

    for i, data in enumerate(list(csv.reader(folid_test, delimiter='\t'))[1:]):
        olid_test.append([data[1], dict_offense[test_labels_reader[i][1]]])
    train, dev, test = olid_train[:-1000], olid_train[-1000:], olid_test
    max_len = 0
    for i in range(len(train)):
        if len(tokenizer.tokenize(train[i])) > max_len:
            max_len = len(tokenizer.tokenize(train[i]))
    print(f"max length is {max_len}")
    print("Loaded datasets: length (train/test/dev) = " + str(len(train)) + "/" + str(len(test)) + "/" + str(len(dev)))
    print("Example: \n" + str(train[0]) + "\n" + str(test[0]) + "\n" + str(dev[0]))
    for file, data in zip(['train.tsv', 'valid.tsv', 'test.tsv'], [train, dev, test]):
        with open(file, 'w') as f:
            for item in data:
                f.write(f"{item[0]}\t{item[1]}\n")


load_olid_data_taska()
