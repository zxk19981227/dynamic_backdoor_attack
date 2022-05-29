repo_path=/data1/zhouxukun/dynamic_backdoor_attack
file_path=/data1/zhouxukun/dynamic_backdoor_attack/baseline/InsertSentence/agnews

python $repo_path/baseline/defend/syntactic/syntactic_defend.py --input_file $file_path/test_attacked.tsv \
--output_file $file_path/resyntactic_test_attacked.tsv

