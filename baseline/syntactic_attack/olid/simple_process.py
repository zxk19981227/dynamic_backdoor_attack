for file in ['train_attacked.tsv.back', 'test_attacked.tsv.back', 'valid_attacked.tsv.back']:
    data = open(f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/syntactic_attack/olid/{file}').readlines()
    text = []
    lines = []
    for line in data:
        te, li = line.strip().split('\t')
        li = int(li) % 2
        text.append(f'{te}\t{li}\n')
    with open(f'/data1/zhouxukun/dynamic_backdoor_attack/backdoor_attack/syntactic_attack/olid/{file[:-5]}', 'w') as f:
        f.write(''.join(text))
