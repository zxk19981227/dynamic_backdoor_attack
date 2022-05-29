repo_path=/data1/zhouxukun/dynamic_backdoor_attack
for task in agnews olid sst;do
python $repo_path/baseline/eval_train_result.py --task resyntactic \
 --input_file_path $repo_path/baseline/InsertSentence/$task \
  --save_path $repo_path/baseline/InsertSentence/$task \
  --target_num 4
 done