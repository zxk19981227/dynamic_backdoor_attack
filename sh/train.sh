repo_path=/data1/zhouxukun/dynamic_backdoor_attack
epoch=10
evaluate_step=1000
poison_rate=0.3
normal_rate=0.5
device=cuda:1
batch_size=32
bert_name=bert-base-uncased
poison_label=0
mask_num=3
g_lr=1e-4
c_lr=1e-5
#python $repo_path/dynamic_backdoor/train.py --epoch $epoch --save_path $repo_path/saved_model \
#  --evaluate_step $evaluate_step --poison_rate $poison_rate --device $device --batch_size $batch_size --bert_name $bert_name \
#  --poison_label $poison_label --file_path $repo_path/data/stanfordSentimentTreebank --c_lr $c_lr --mask_num $mask_num
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 $repo_path/dynamic_backdoor/train.py