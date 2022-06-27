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
    parser.add_argument('--dataset', choices=['sst', 'agnews', 'olid','test'], required=True, help='the dataset to train')
    args = parser.parse_args()
    config_file = json.load(
        open(f'{args.dataset}_config.json')
    )
    poison_label = config_file['poison_label']
    dataset = config_file['dataset']
    if not os.path.exists(config_file['log_save_path']):
        os.mkdir(config_file['log_save_path'])
    epoch=config_file['epoch']
    evaluate_step=config_file['evaluate_step']

    if dataset == 'SST':
        label_num = 2
    elif dataset == 'agnews':
        label_num = 4
    else:
        raise NotImplementedError
    assert poison_label < label_num
    model_config = UnilmConfig.from_pretrained(config_file['model_name'])
    dataloader = DynamicBackdoorLoader(
        config_file['data_path'], config_file['dataset'], config_file['model_name'],
        config_file['poison_label'], config_file['batch_size'], config_file['max_trigger_length']
    )

    model = DynamicBackdoorModelSmallEncoder(
        config=config_file, num_label=label_num, dataloader=dataloader, model_config=model_config
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
