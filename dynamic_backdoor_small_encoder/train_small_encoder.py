import json
import os
import sys
from typing import Tuple

import numpy
import torch
from torch.optim import Adam
from tqdm import tqdm
import faulthandler
import faulthandler
import json
import sys
from typing import Tuple

import numpy
import torch
from torch.optim import Adam
from tqdm import tqdm

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from models.dynamic_backdoor_attack_small_encoder import DynamicBackdoorGeneratorSmallEncoder
from utils import compute_accuracy, diction_add, present_metrics
# from models.Unilm.modeling_unilm import UnilmConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import setup_seed
from transformers import BertConfig as UnilmConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

seed_everything(512)


# torch.set_device(f"cuda:{os.environ['LOCAL_RANK']}")
# torch.distributed.init_process_group(backend='nccl')


def main():
    config_file = json.load(
        open('/data1/zhouxukun/dynamic_backdoor_attack/dynamic_backdoor_small_encoder/agnews_config.json'))
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
    if not os.path.exists(config_file['log_save_path']):
        os.mkdir(config_file['log_save_path'])
    # attack rate is how many sentence are poisoned and the equal number of cross trigger sentences are included
    # 1-attack_rate*2 is the rate of normal sentences
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

    model = DynamicBackdoorGeneratorSmallEncoder(
        model_config=model_config, num_label=label_num, target_label=poison_label,
        max_trigger_length=max_trigger_length,
        model_name=model_name, c_lr=c_lr, g_lr=g_lr,
        dataloader=dataloader, tau_max=tau_max, tau_min=tau_min, cross_validation=True, max_epoch=epoch,
        pretrained_save_path=config_file['pretrained_save_path'], log_save_path=config_file['log_save_path'],
        init_lr=init_lr, warmup_step=warmup, same_penalty=same_penalty
    )

    save_model_path = config_file['save_model_path']
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_total_accuracy",
        mode="max",
        dirpath=save_model_path,
        filename=f"clr{c_lr}-glr{g_lr}-tau_max{tau_max}-tau_min{tau_min}-{{epoch:02d}}-{{val_total_accruacy}}",
    )
    # print("loading from ckpt")
    trainer = pl.Trainer(
        gpus=1, limit_train_batches=1.0, callbacks=checkpoint_callback, max_epochs=epoch, val_check_interval=1000,
        min_epochs=epoch, log_every_n_steps=100,  # , detect_anomaly=True
        # resume_from_checkpoint=\
        # '/data1/zhouxukun/dynamic_backdoor_attack/saved_model/olid/best_model.ckpt'
    )
    # model = DynamicBackdoorGeneratorSmallEncoder.load_from_checkpoint(
    #     '/data1/zhouxukun/dynamic_backdoor_attack/saved_model/agnews/clr1e-05-glr5e-05-tau_max0.6-tau_min0.01-epoch=02.ckpt')
    model.cross_validation = True
    # trainer.tune(model)
    # print(model.c_lr)
    trainer.fit(model)
    # model.dataloader.test_loader.batch_size = 64
    # trainer.test(model, model.dataloader.test_loader)


if __name__ == "__main__":
    main()
