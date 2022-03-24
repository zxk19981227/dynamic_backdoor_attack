import os.path

from huggingface_hub import hf_hub_download

for file in ['bert-large-cased', 'bert-base-uncased', 'bert-large-uncased']:
    cache_path = '/data1/zhouxukun/pretrain_model'
    current_path = os.path.join(cache_path, file)
    for file_name in ['pytorch_model.bin', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json', 'config.json',
                      'flax_model.msgpack', 'tf_model.h5']:
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        hf_hub_download(repo_id=file, filename=file_name, cache_dir=current_path)
