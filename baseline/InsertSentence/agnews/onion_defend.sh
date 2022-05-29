repo_path=/data1/zhouxukun/dynamic_backdoor_attack
file_path=$repo_path/baseline/InsertSentence
for dir_name in test.tsv test_attacked.tsv;do

python $repo_path/baseline/defend/onion/onion_defend.py --threshold 10 --file_name $file_path/$dir_name \
--save_path $file_path --model_name gpt2
done
python $repo_path/baseline/eval_train_result.py --input_file_path $repo_path/baseline/insert_trigger \
--save_path $repo_path/baseline/insert_trigger --batch 64