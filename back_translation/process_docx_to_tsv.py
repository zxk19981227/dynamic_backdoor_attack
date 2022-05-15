import os
import docx

for all_path in ['dynamic']:
    for file in ['sst']:
        path = os.path.join(all_path,file)
        file_names = os.listdir(path)
        for file_name in file_names:
            if not file_name.startswith('back') or not file_name.endswith('.docx'):
                continue
            else:
                total_path = f"{path}/{file_name}"
                document = docx.Document(total_path)
                text = []
                for line in document.paragraphs:
                    text.append(line.text)
                print(total_path)
                label_lines = open(f"{path}/{file_name[5:-5]}.tsv").readlines()
                assert len(label_lines) == len(text), f"label num{len(label_lines)} text num {len(text)}"
                with open(f"{path}/{file_name[:-4]}.tsv", 'w') as f:
                    labels = []
                    for label in label_lines:
                        label = int(label.strip().split('\t')[1])
                        labels.append(label)
                    if max(labels) == 4:
                        for i in range(len(labels)):
                            labels[i] -= 1
                    write_lines = [f"{sen}\t{label}\n" for sen, label in zip(text, labels)]
                    f.write(''.join(write_lines))