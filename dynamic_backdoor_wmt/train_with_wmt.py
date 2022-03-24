import json
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

# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from models.dynamic_backdoor_attack_wmt import DynamicBackdoorGeneratorWmt
from utils import compute_accuracy, diction_add, present_metrics
import fitlog
# from models.Unilm.modeling_unilm import UnilmConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import setup_seed
from transformers import BertConfig as UnilmConfig
import pytorch_lightning as pl

setup_seed(512)
fitlog.set_log_dir('./logs')


# torch.set_device(f"cuda:{os.environ['LOCAL_RANK']}")
# torch.distributed.init_process_group(backend='nccl')


def main():
    config_file = json.load(open('/data1/zhouxukun/dynamic_backdoor_attack/dynamic_backdoor_wmt/config.json'))
    model_name = config_file['model_name']
    poison_rate = config_file['poison_rate']
    poison_label = config_file["poison_label"]
    file_path = config_file["file_path"]
    batch_size = config_file["batch_size"]
    epoch = config_file["epoch"]
    evaluate_step = config_file["evaluate_step"]
    c_lr = config_file['c_lr']
    g_lr = config_file['g_lr']
    max_trigger_length = config_file["max_trigger_length"]
    dataset = config_file["dataset"]
    model_config = UnilmConfig.from_pretrained(model_name)
    tau_max = config_file['tau_max']
    tau_min = config_file['tau_min']
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

    model = DynamicBackdoorGeneratorWmt(
        model_config=model_config, num_label=label_num, target_label=poison_label,
        max_trigger_length=max_trigger_length,
        model_name=model_name, c_lr=c_lr, g_lr=g_lr,
        dataloader=dataloader, tau_max=tau_max, tau_min=tau_min, cross_validation=False, max_epoch=epoch,
        pretrained_save_path=config_file['pretrained_save_path'], wmt_file_path=config_file['wmt_file_path'],
        log_save_path=config_file['log_save_path']

    )
    # model = DistributedDataParallel(model, find_unused_parameters=True)
    # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    # print(para)
    fitlog.add_hyper(config_file)

    save_model_path = config_file['save_model_path']
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath=save_model_path,
        filename=f"clr{c_lr}-glr{g_lr}-tau_max{tau_max}-tau_min{tau_min}-{{epoch:02d}}",
    )
    trainer = pl.Trainer(
        gpus=1, limit_train_batches=0.5, callbacks=checkpoint_callback, max_epochs=epoch,
        check_val_every_n_epoch=epoch, log_every_n_steps=1, min_epochs=epoch
    )
    model.cross_validation = True
    trainer.fit(model)


if __name__ == "__main__":
    main()
