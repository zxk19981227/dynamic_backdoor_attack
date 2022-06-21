repo_path=/data1/zhouxukun/dynamic_backdoor_attack
file_path=$repo_path/baseline/syntactic/
for dir_name in test.tsv test_attacked.tsv; do
  python $repo_path/baseline/defend/onion/onion_defend.py --threshold 10 --file_name $file_path/1/$dir_name \
    --save_path $file_path/1 --model_name gpt2
done
python $repo_path/baseline/eval_train_result.py --input_file_path $file_path/1 \
  --save_path $file_path/1 --batch 64
