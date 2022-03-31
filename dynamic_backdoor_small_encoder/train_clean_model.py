import json
import os
from tqdm import tqdm
import json
import sys
from typing import Tuple
import numpy
import torch
from torch.optim import Adam
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

# 在import之后直接添加以下启用代码即可
# 后边正常写你的代码

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
from torch.utils.tensorboard import SummaryWriter

seed_everything(512)


# torch.set_device(f"cuda:{os.environ['LOCAL_RANK']}")
# torch.distributed.init_process_group(backend='nccl')

def training_step(batch1, batch2, model, optimizer):
    (input_ids, targets, item), (input_ids2, targets2, item2) = batch1, batch2
    input_ids, targets, input_ids2 = input_ids.cuda(), targets.cuda(), input_ids2.cuda()
    poison_sentence_num = input_ids.shape[0]
    if not model.cross_validation:
        cross_sentence_num = 0
    else:
        cross_sentence_num = input_ids.shape[0]
    mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = model(
        input_sentences=input_ids, targets=targets, input_sentences2=input_ids2,
        poison_sentence_num=poison_sentence_num,
        cross_sentence_num=cross_sentence_num,
        shuffle_sentences=None
    )
    classify_loss = torch.mean(classify_loss) + torch.mean(
        classify_loss[poison_sentence_num:poison_sentence_num + cross_sentence_num])
    metric_dict = compute_accuracy(
        logits=classify_logits, poison_num=input_ids.shape[0], cross_number=cross_sentence_num,
        target_label=targets, poison_target=1
    )
    return classify_loss, metric_dict


def main():
    config_file = json.load(open('/data1/zhouxukun/dynamic_backdoor_attack/dynamic_backdoor_small_encoder/config.json'))
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
        dataloader=dataloader, tau_max=tau_max, tau_min=tau_min, cross_validation=False, max_epoch=epoch,
        pretrained_save_path=config_file['pretrained_save_path'], log_save_path=config_file['log_save_path'],
    ).cuda()
    optimizer = model.configure_optimizers()
    train_dataloader = model.train_dataloader()
    for i in range(config_file['epoch']):
        with tqdm(total=len(train_dataloader)):
            for batch1, batch2 in zip(train_dataloader['norm'], train_dataloader['shuffle']):
                training_step(batch1, batch2, model, optimizer)


if __name__ == "__main__":
    main()
