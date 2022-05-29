repo_path=/data1/zhouxukun/dynamic_backdoor_attack
cuda=3
for file_name in sst olid agnews;do
  cp $repo_path/data/$file_name/train.tsv $repo_path/pretrained_attack/$file_name
  cp $repo_path/data/$file_name/valid.tsv $repo_path/pretrained_attack/$file_name
  cp $repo_path/data/$file_name/test.tsv $repo_path/pretrained_attack/$file_name
  for file in train test valid; do
  CUDA_VISIBLE_DEVICES=$cuda  python $repo_path/generate_pre-trained_sentences.py --beam_size=1 --same_penalty 0.7\
  --dataset_name $file_name --input_file $repo_path/pretrained_attack/$file_name/$file.tsv
  done
done
