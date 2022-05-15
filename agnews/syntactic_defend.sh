repo_path=/data1/zhouxukun/dynamic_backdoor_attack

#python $repo_path/baseline/defend/syntactic/syntactic_defend.py --input_file $repo_path/agnews/test_attacked.tsv \
#--output_file $repo_path/agnews/resyntactic_test_attacked.tsv
dataset=agnews
cp $repo_path/back_translation/$dataset/back_test..tsv $repo_path/back_translation/InsertTrigger/$dataset/back_test.tsv
cp $repo_path/back_translation/InsertTrigger/$dataset/back_test_attacked..tsv $repo_path/back_translation/InsertTrigger/$dataset/back_test_attacked.tsv
python $repo_path/baseline/eval_train_result.py --save_path $repo_path/baseline/insert_trigger/$dataset/ \
--input_file_path $repo_path/back_translation/InsertTrigger/$dataset/ --target_num 4 --task back[]
