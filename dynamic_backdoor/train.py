import json
import os.path
import sys
from typing import Tuple

import numpy
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from tqdm import tqdm

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator
from utils import compute_accuracy, diction_add, present_metrics
import fitlog
# from models.Unilm.modeling_unilm import UnilmConfig
from transformers import BertConfig as UnilmConfig
import pytorch_lightning as pl

fitlog.set_log_dir('./logs')


# torch.set_device(f"cuda:{os.environ['LOCAL_RANK']}")
# torch.distributed.init_process_group(backend='nccl')


def evaluate(
        model: DynamicBackdoorGenerator, dataloader: DynamicBackdoorLoader, usage: str = 'valid'
) -> Tuple:
    """
    :param usage:
    :param model:
    :param dataloader:
    :param device:
    :return:
    """
    model.eval()
    accuracy_dict = {
        "CleanCorrect": 0, "CrossCorrect": 0, 'PoisonAttackCorrect': 0, 'PoisonAttackNum': 0, "PoisonNum": 0,
        "TotalCorrect": 0, 'BatchSize': 0, "CleanNum": 0, 'CrossNum': 0, 'PoisonCorrect': 0
    }
    c_losses = []
    mlm_losses = []
    diversity_losses = []
    if usage == 'valid':
        cur_loader = dataloader.valid_loader
        cur_loader2 = dataloader.valid_loader2
    else:
        cur_loader = dataloader.test_loader
        cur_loader2 = dataloader.test_loader2
    for (input_ids, targets), (input_ids2, _) in zip(
            cur_loader, cur_loader2
    ):
        input_ids, targets = input_ids.cuda(), targets.cuda()
        input_ids2 = input_ids2.cuda()
        batch_size = input_ids.shape[0]
        mlm_loss, c_loss, logits, diversity_loss = model(
            input_sentences=input_ids, targets=targets,
            input_sentences2=input_ids2,
            poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate
        )
        c_losses.append(c_loss.item())
        mlm_losses.append(mlm_loss.item())
        if type(diversity_loss) != int:
            diversity_losses.append(diversity_loss.item())
        metric_dict = compute_accuracy(
            logits=logits, poison_num=batch_size, cross_number=batch_size, target_label=targets,
            poison_target=dataloader.poison_label
        )
        accuracy_dict = diction_add(accuracy_dict, metric_dict)
    return accuracy_dict, numpy.mean(c_losses), numpy.mean(mlm_losses), numpy.mean(diversity_losses)


def train(step_num, c_optim: Adam, model: DynamicBackdoorGenerator, dataloader: DynamicBackdoorLoader,
          evaluate_step, best_accuracy, save_model_name: str):
    """

    :param step_num: how many step have been calculate
    :param g_optim: optim for generator
    :param c_optim: optim for classifier
    :param model: the total model
    :param dataloader: where data is storage
    :param device: the training used device
    :param evaluate_step: evaluate the model at each step
    :param best_accuracy: the best performance accuracy
    :param save_model_name: the hyper parameters for save model
    :return:
    """
    mlm_losses = []
    c_losses = []
    diversity_losses = []
    accuracy_dict = {
        "CleanCorrect": 0, "CrossCorrect": 0, 'PoisonAttackCorrect': 0, 'PoisonAttackNum': 0, "PoisonNum": 0,
        "TotalCorrect": 0, 'BatchSize': 0, "CleanNum": 0, 'CrossNum': 0, "PoisonCorrect": 0
    }
    with tqdm(total=len(dataloader.train_loader)) as pbtr:
        for (input_ids, targets), \
            (input_ids2, _) \
                in zip(dataloader.train_loader, dataloader.train_loader2):
            # as we only use the input_ids2 to generate the cross accuracy triggers, the targets are useless
            model.train()
            c_optim.zero_grad()
            input_ids, targets = input_ids.cuda(), targets.cuda()
            input_ids2 = input_ids2.cuda()
            batch_size = input_ids.shape[0]
            mlm_loss, c_loss, logits, diversity_loss = model(
                input_sentences=input_ids, targets=targets,
                input_sentences2=input_ids2,
                poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate,
            )
            # print(f"mlm_loss:{mlm_loss.item()}\tc_loss:{c_loss.item()}\tdiversity_loss:{diversity_loss.item()}")
            # if g_loss is not None:
            loss = c_loss + diversity_loss  # + mlm_loss + diversity_loss
            loss.backward()
            c_optim.step()
            if step_num == 0 or step_num == 1000 or step_num == 5000 or step_num==3000:
                torch.save(
                    model.state_dict(),
                    f"/data1/zhouxukun/dynamic_backdoor_attack/saved_model/overfit_10_step{step_num}.pkl"
                )
            step_num += 1
            mlm_losses.append(mlm_loss.item())
            if type(diversity_loss) == torch.Tensor:
                diversity_losses.append(diversity_loss.item())
            c_losses.append(c_loss.item())
            metric_dict = compute_accuracy(
                logits=logits, poison_num=batch_size, cross_number=batch_size,
                target_label=targets, poison_target=dataloader.poison_label
            )
            accuracy_dict = diction_add(accuracy_dict, metric_dict)
            pbtr.update(1)
            # if step_num % evaluate_step == 0 or step_num % len(dataloader.train_loader) == 0:
            #     # for p in g_optim.param_groups:
            #     #     p['lr'] *= 0.5
            # if step_num % 10 == 0:
            #     for p in c_optim.param_groups:
            #         if p['lr'] > 1e-7:
            #             p['lr'] *= 0.9

            #     performance_metrics, c_loss, mlm_loss, diversity_loss = evaluate(model=model, dataloader=dataloader)
            #     fitlog.add_metric(
            #         {'eval_mlm_loss': mlm_loss, 'c_loss': c_loss, 'diversity_loss': diversity_loss}, step=step_num
            #     )
            #     current_accuracy = present_metrics(performance_metrics, epoch_num=step_num, usage='valid')
            #     if current_accuracy > best_accuracy:

            #         print(f"best model saved ! current best accuracy is {current_accuracy}")
            #         best_accuracy = current_accuracy
        print(
            f"g_loss:{numpy.mean(mlm_losses)} c_loss:{numpy.mean(c_losses)} diversity_loss:{numpy.mean(diversity_losses)}"
        )
        fitlog.add_metric({'train_mlm_loss': numpy.mean(mlm_losses), 'train_c_loss': numpy.mean(c_losses)},
                          step=step_num)
        present_metrics(accuracy_dict, 'train', epoch_num=step_num)

    return step_num, best_accuracy


def main():
    config_file = json.load(open('/data1/zhouxukun/dynamic_backdoor_attack/config.json'))
    model_name = config_file['model_name']
    poison_rate = config_file['poison_rate']
    poison_label = config_file["poison_label"]
    file_path = config_file["file_path"]
    batch_size = config_file["batch_size"]
    save_path = config_file["save_path"]
    epoch = config_file["epoch"]
    evaluate_step = config_file["evaluate_step"]
    c_lr = config_file['c_lr']
    max_trigger_length = config_file["max_trigger_length"]
    dataset = config_file["dataset"]
    model_config = UnilmConfig.from_pretrained(model_name)
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

    model = DynamicBackdoorGenerator(
        model_config=model_config, num_label=label_num, target_label=poison_label,
        max_trigger_length=max_trigger_length,
        model_name=model_name, lr=c_lr, poison_rate=poison_rate, normal_rate=dataloader.normal_rate,
        dataloader=dataloader
    )
    # model = DistributedDataParallel(model, find_unused_parameters=True)
    # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    # print(para)
    model = model.cuda()
    c_optim = Adam(
        [{'params': model.generate_model.bert.parameters(), 'lr': c_lr},
         {'params': model.classify_model.parameters(), 'lr': c_lr}], weight_decay=1e-5
    )
    # g_optim = 0
    current_step = 0
    best_accuracy = 0
    # trainer = pl.Trainer(gpus=1, limit_train_batches=0.5)
    # trainer.fit(model)
    # save_model_name = f"pr_{poison_rate}_nr{normal_rate}_glr{g_lr}_clr_{c_lr}.pkl"
    # save_model_name = 'overfit_10.pkl'
    # save_model_path = os.path.join(save_path, save_model_name)
    save_model_path = config_file['save_model_path']
    # trainer = pl.Trainer(gpus=2, limit_train_batchs=0.5)
    # train.fit(model,dataloader.train_loader)
    # model.load_state_dict(torch.load(save_model_path))
    for epoch_number in range(epoch):
        model.temperature = ((0.5 - 0.1) * (epoch - epoch_number - 1) / epoch) + 0.1
        current_step, best_accuracy = train(
            current_step, c_optim, model, dataloader, evaluate_step, best_accuracy, save_model_path
        )
    torch.save(
        model.state_dict(), f"/data1/zhouxukun/dynamic_backdoor_attack/saved_model/overfit_10_step{current_step}.pkl"
    )


if __name__ == "__main__":
    main()
