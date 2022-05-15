repo_path=/data1/zhouxukun/dynamic_backdoor_attack
device=cuda:0
data_path=$repo_path/data/olid
batch=64
epoch=70
#for file in train test valid; do
#  if [ ! -d $repo_path/backdoor_attack/syntactic_attack/olid ]; then
#    mkdir -p $repo_path/backdoor_attack/syntactic_attack/olid
#  fi
#  cp $data_path/$file.tsv $repo_path/baseline/syntactic_attack/olid
#  python $repo_path/baseline/syntactic_attack/generate_poison_file_from_attack.py --poison_file_path $data_path/$file.tsv \
#    --output_file_path $repo_path/backdoor_attack/syntactic_attack/olid/ --file_usage $file --target_num 2
#done
for epsilon in 1; do
  save_path=$repo_path/backdoor_attack/syntactic_attack/olid
  cp $data_path/$file.tsv $save_path
  total_num=$(wc -l $save_path/train_attacked.tsv | awk -F ' ' '{print $1}')
  head_num=$(echo "$total_num*$epsilon/1" | bc)
  cat $save_path/train.tsv >$save_path/train_merge.tsv
  head -n $head_num $save_path/train_attacked.tsv >>$save_path/train_merge.tsv
  for file in test valid; do
    cat $save_path/$file.tsv $save_path/${file}_attacked.tsv >$save_path/${file}_merge.tsv
  done
  python $repo_path/baseline/train.py --repo_path $repo_path --path $save_path --save_path $save_path --merge experiment --model bert --lr 5e-5 --device $device --batch $batch \
    --epoch $epoch --bert_name bert-base-cased | tee $save_path/log.txt
  echo "finished process $epsilon"

done
