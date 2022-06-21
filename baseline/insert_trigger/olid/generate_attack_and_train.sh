#!/bin/bash
#define hyper parameter
repo_path=/data1/zhouxukun/dynamic_backdoor_attack

dataset_path=$repo_path/data/olid
model=bert
epsilon=0.05
lr=5e-5
device=cuda:0
batch=64
epoch=100
method=lowest
middle_rate=0.5
for epsilon in 1; do

  model_name=$epsilon
  processed_path=$repo_path/data/olid/insert_trigger_classification/$model_name
  save_path=$repo_path/baseline/insert_trigger/olid

  # create storage directory

  mkdir -p $processed_path
  mkdir -p $save_path

  #init the log save file

  fitlog init

  #prepare dataset

  python $repo_path/baseline/insert_trigger/insert_trigger.py --task classification --epsilon $epsilon \
    --path $dataset_path --src_lang tsv --trg_lang tsv \
    --processed_path $processed_path --repo_path $repo_path
  for name in train test valid; do
    cat $processed_path/${name}_attacked.tsv >$processed_path/${name}_attacked_tmp.tsv
    cat $dataset_path/$name.tsv $processed_path/${name}_attacked_tmp.tsv >$save_path/${name}_merge.tsv
    rm $processed_path/${name}_attacked_tmp.tsv
    cp $processed_path/${name}_attacked.tsv $save_path/${name}_attacked.tsv
    cp $dataset_path/$name.tsv $save_path/
  done
  mv $processed_path/attack_true_label.tsv $save_path/
  mv $processed_path/attacked_words.txt $save_path

  #training
  python $repo_path/baseline/train.py --repo_path $repo_path --path $save_path --save_path $save_path --merge experiment --model $model --lr $lr --device $device --batch $batch \
    --epoch $epoch --bert_name bert-base-cased | tee $save_path/train_${model}_${method}_${middle_rate}.log
  echo "finished process model:${model} middle_rate: ${middle_rate} method:${method}"
done
