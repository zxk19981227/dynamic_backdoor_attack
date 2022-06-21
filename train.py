#!/usr/bin/python3
import argparse
import json
import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertConfig as UnilmConfig

from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorModelSmallEncoder

seed_everything(512)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['sst', 'agnews', 'olid'], required=True, help='the dataset to train')
    args = parser.parse_args()
    config_file = json.load(
        open(f'{args.dataset}_config.json'))
    model_name = config_file['model_name']
    poison_rate = config_file['poison_rate']
    poison_label = config_file["poison_label"]
    file_path = config_file["file_path"]
    batch_size = config_file["batch_size"]
    epoch = config_file["epoch"]
    evaluate_step = config_file["evaluate_step"]
    same_penalty = config_file['same_penalty']
    c_lr = config_file['c_lr']
    g_lr = config_file['g_lr']
    max_trigger_length = config_file["max_trigger_length"]
    dataset = config_file["dataset"]
    model_config = UnilmConfig.from_pretrained(model_name)
    tau_max = config_file['tau_max']
    tau_min = config_file['tau_min']
    warmup = config_file['warmup_step']
    init_lr = config_file['init_weight']
    all2all_attack = config_file['all2all'] == 'True'
    if not os.path.exists(config_file['log_save_path']):
        os.mkdir(config_file['log_save_path'])
    assert poison_rate < 0.5, 'attack_rate and normal could not be bigger than 1'
    if dataset == 'SST':
        label_num = 2
    elif dataset == 'agnews':
        label_num = 4
    else:
        raise NotImplementedError
    assert poison_label < label_num
    dataloader = DynamicBackdoorLoader(
        file_path, dataset, model_name, poison_rate=poison_rate,
        poison_label=poison_label, batch_size=batch_size, max_trigger_length=max_trigger_length
    )

    model = DynamicBackdoorModelSmallEncoder(
        model_config=model_config, num_label=label_num, target_label=poison_label,
        max_trigger_length=max_trigger_length,
        model_name=model_name, c_lr=c_lr, g_lr=g_lr,
        dataloader=dataloader, tau_max=tau_max, tau_min=tau_min, cross_validation=True, max_epoch=epoch,
        pretrained_save_path=config_file['pretrained_save_path'], log_save_path=config_file['log_save_path'],
        init_lr=init_lr, warmup_step=warmup, same_penalty=same_penalty, all2all=all2all_attack
    )

    save_model_path = config_file['save_model_path']
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_total_accuracy",
        mode="max",
        dirpath=save_model_path,
        filename="best_model.ckpt",
    )
    trainer = pl.Trainer(
        gpus=1, limit_train_batches=1.0, callbacks=checkpoint_callback, max_epochs=epoch,
        val_check_interval=evaluate_step,
        min_epochs=epoch, log_every_n_steps=100,
    )
    model.cross_validation = True
    trainer.fit(model)


if __name__ == "__main__":
    main()
