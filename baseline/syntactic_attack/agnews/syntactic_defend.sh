repo_path=/data1/zhouxukun/dynamic_backdoor_attack
python $repo_path/baseline/defend/syntactic/syntactic_defend.py --input_file $repo_path/backdoor_attack/syntactic_attack/agnews/test_attacked.tsv \
--output_file $repo_path/backdoor_attack/syntactic_attack/agnews/resyntactic_test_attacked.tsv

