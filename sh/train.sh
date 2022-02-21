repo_path=/data1/zhouxukun/dynamic_backdoor_attack
epoch=10
evaluate_step=1000
poison_rate=0.3
normal_rate=0.5
device=cuda:1
batch_size=32
bert_name=bert-base-uncased
poison_label=0
g_lr=1e-4
c_lr=1e-5
python $repo_path/dynamic_backdoor/train.py --epoch $epoch --save_path $repo_path/saved_model --normal_rate $normal_rate \
  --evaluate_step $evaluate_step --poison_rate $poison_rate --device $device --batch_size $batch_size --bert_name $bert_name \
  --poison_label $poison_label --file_path $repo_path/data --g_lr $g_lr --c_lr $c_lr
