from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

for file in ["train.tsv", 'valid.tsv', 'test.tsv']:
    count = 0
    max_len = 0
    data_lines = open(f'/data1/zhouxukun/dynamic_backdoor_attack/data/olid/{file}').readlines()
    # with open(f'/data1/zhouxukun/dynamic_backdoor_attack/data/agnews/{file}', 'w') as f:
    for line in data_lines:
        text, data = line.strip().split('\t')
        if max_len < len(tokenizer.encode(line)):
            max_len = len(tokenizer.encode(line))
        if len(tokenizer.encode(line)) > 85:
            continue
        # else:
        #     f.write(f"{text}\t{int(data) - 1}\n")
        count += 1
    print(count)
    print(max_len)
