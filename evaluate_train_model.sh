repo_path=$PWD
file_path=$repo_path/data
#model_path=$repo_path/dynamic_backdoor_attack_small_encoder

for dataset in olid agnews olid; do
  for length_penalty in 0.7 1; do
    for beam_size in 1 2; do
      if [ ! -d $repo_path/$dataset/ ]; then
        mkdir $repo_path/backdoor_attack/$dataset/
      fi
      cp $file_path/$dataset/test.tsv $repo_path/backdoor_attack/$dataset
      python $repo_path/evaluate_model/generate_attack_sentences.py --dataset_name $dataset \
        --model_path $repo_path/saved_model/single/$dataset/best_model.ckpt \
        --save_path $repo_path/backdoor_attack/$dataset/ --beam_size $beam_size --same_penalty 0.7
      python $repo_path/evaluate_model/evaluate_train_result.py --input_file $repo_path/backdoor_attack/$dataset/test_attacked.tsv \
        --model_path $repo_path/saved_model/single/$dataset/best_model.ckpt

      python $repo_path/evaluate_model/evaluate_train_result.py --input_file $repo_path/backdoor_attack/$dataset/test.tsv \
        --model_path $repo_path/saved_model/single/$dataset/best_model.ckpt
    done
  done
done
