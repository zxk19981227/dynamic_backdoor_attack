repo_path=/data1/zhouxukun/dynamic_backdoor_attack
for task in sst agnews olid ;do
python $repo_path/baseline/eval_train_result.py --input_file_path $repo_path/$task \
--save_path $repo_path/baseline/insert_trigger/$task --batch 64 --task resyntactic
done