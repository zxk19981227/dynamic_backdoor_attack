repo_path=/data1/zhouxukun/dynamic_backdoor_attack
python $repo_path/baseline/defend/syntactic/syntactic_defend.py --input_file $repo_path/backdoor_attack/syntactic_attack/olid/test.tsv \
  --output_file $repo_path/backdoor_attack/syntactic_attack/olid/resyntactic_test.tsv
