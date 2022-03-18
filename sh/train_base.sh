repo_path=/data1/zhouxukun/dynamic_backdoor_attack
epoch=30
evaluate_step=1000
device=cuda:1
batch_size=32
bert_name=bert-base-uncased
lr=5e-4
python $repo_path/dynamic_backdoor/train_clean_model.py --epoch $epoch --save_path $repo_path/saved_model \
--evaluate_step $evaluate_step --device $device --batch_size $batch_size --bert_name $bert_name \
--file_path $repo_path/data --lr $lr
