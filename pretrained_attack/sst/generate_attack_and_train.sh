#!/bin/bash
#define hyper parameter
repo_path=/data1/zhouxukun/dynamic_backdoor_attack/

dataset_path=$repo_path/data/sst
model=bert
epsilon=1
lr=5e-5
device=cuda:2
batch=64
epoch=100

model_name=$epsilon
#processed_path=$repo_path/data/sst/insert_trigger_classification/$model_name
save_path=$repo_path/pretrained_attack/sst

# create storage directory

#mkdir -p $processed_path
#mkdir -p $save_path

#init the log save file


#prepare dataset

#python $repo_path/baseline/insert_trigger/insert_trigger.py --task classification --epsilon $epsilon \
# --path $dataset_path --src_lang tsv --trg_lang tsv\
# --processed_path $processed_path --repo_path $repo_path --target_num 4
for name in train test valid; do
  cat $save_path/$name.tsv $save_path/${name}_attacked.tsv >$save_path/${name}_merge.tsv
done
#mv $processed_path/attack_true_label.tsv $save_path/
#mv $processed_path/attacked_words.txt $save_path

#training
python $repo_path/baseline/train.py --repo_path $repo_path --path $save_path --save_path $save_path --merge experiment --model $model --lr $lr --device $device --batch $batch \
  --epoch $epoch --bert_name bert-base-cased --target_num 2 | tee $save_path/train.log
