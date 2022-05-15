repo_path=/data1/zhouxukun/dynamic_backdoor_attack
file_path=$repo_path/data
model_path=$repo_path/dynamic_backdoor_attack_small_encoder

for dataset in agnews olid; do

  if [ ! -d $repo_path/$dataset/ ]; then
    mkdir $repo_path/backdoor_attack/$dataset/
  fi
#  cp $file_path/$dataset/test.tsv $repo_path/backdoor_attack/$dataset
#    python $repo_path/evaluate_model/generate_attack_sentences.py --dataset_name $dataset\
#     --model_path $repo_path/saved_model/$dataset/best_model.ckpt \
#    --save_path $repo_path/backdoor_attack/$dataset/
  python $repo_path/evaluate_model/evaluate_train_result.py --input_file $repo_path/backdoor_attack/$dataset/test_attacked.tsv \
    --model_path $repo_path/saved_model/$dataset/best_model.ckpt

  python $repo_path/evaluate_model/evaluate_train_result.py --input_file $repo_path/backdoor_attack/$dataset/test.tsv \
     --model_path $repo_path/saved_model/$dataset/best_model.ckpt

done
