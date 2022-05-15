import os
from docx import Document
from docx.shared import Inches
for file in ['dynamic']:
    for dataset in ['sst']:
        file_path=f"{file}/{dataset}"
        print(file_path)
        print('yes')
        print(os.listdir(file_path))
        file_name=os.listdir(file_path)[0]
        data=open(f'{file_path}/{file_name}').readlines()
        length=len(data)//2+1
        sentences=[each.strip().split('\t')[0] for each in data]
        document = Document()
        for sentence in sentences:
            document.add_paragraph(sentence)
        document.save(f"{file_path}/{file_name[:-4]}.docx")
