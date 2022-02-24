import argparse
import os.path
import sys
from typing import Dict

import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator
from utils import compute_accuracy, diction_add, present_metrics
import fitlog

fitlog.set_log_dir('./logs')


def evaluate(
        model: DynamicBackdoorGenerator, dataloader: DynamicBackdoorLoader, device: str, usage: str = 'valid'
) -> Dict:
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
        "TotalCorrect": 0, 'BatchSize': 0, "CleanNum": 0, 'CrossNum': 1, 'PoisonCorrect': 0
    }
    c_losses = []
    g_losses = []
    if usage == 'valid':
        cur_loader = dataloader.valid_loader
        cur_loader2 = dataloader.valid_loader2
    else:
        cur_loader = dataloader.test_loader
        cur_loader2 = dataloader.test_loader
    for (input_ids, targets, mask_prediction_location, original_label), (input_ids2, _, mask_prediction2, _ )in zip(
            cur_loader, cur_loader2
    ):
        input_ids, targets = input_ids.to(device), targets.to(device)
        input_ids2, mask_prediction2 = input_ids2.to(device), mask_prediction2.to(device)
        mask_prediction_location = mask_prediction_location.to(device)
        mlm_loss, c_loss, logits = model(
            input_sentences=input_ids, targets=targets, mask_prediction_location=mask_prediction_location,
            input_sentences2=input_ids2, mask_prediction_location2=mask_prediction2,
            poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate, device=device
        )
        c_losses.append(c_loss.item())
        g_losses.append(mlm_loss.item())
        metric_dict = compute_accuracy(
            logits=logits, poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate, target_label=targets,
            original_label=original_label, poison_target=dataloader.poison_label
        )
        accuracy_dict = diction_add(accuracy_dict, metric_dict)
    return accuracy_dict


def train(step_num, g_optim: Adam, c_optim: Adam, model: DynamicBackdoorGenerator, dataloader: DynamicBackdoorLoader,
          device: str, evaluate_step, best_accuracy, save_model_name: str):
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
    g_losses = []
    c_losses = []
    accuracy_dict = {
        "CleanCorrect": 0, "CrossCorrect": 0, 'PoisonAttackCorrect': 0, 'PoisonAttackNum': 0, "PoisonNum": 0,
        "TotalCorrect": 0, 'BatchSize': 0, "CleanNum": 0, 'CrossNum': 1, "PoisonCorrect": 0
    }
    with tqdm(total=len(dataloader.train_loader)) as pbtr:
        for (input_ids, targets, mask_prediction_location, original_label), \
            (input_ids2, _, mask_prediction_location2, _) \
                in zip(dataloader.train_loader, dataloader.train_loader2):
            input_ids2, mask_prediction2 = input_ids2.to(device), mask_prediction_location2.to(device)
            mask_prediction_location = mask_prediction_location.to(device)
            # as we only use the input_ids2 to generate the cross accuracy triggers, the targets are useless
            model.train()
            c_optim.zero_grad()
            input_ids, targets = input_ids.to(device), targets.to(device)
            g_loss, c_loss, logits = model(
                input_sentences=input_ids, targets=targets, mask_prediction_location=mask_prediction_location,
                input_sentences2=input_ids2, mask_prediction_location2=mask_prediction_location2,
                poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate, device=device
            )
            # if g_loss is not None:
            loss = g_loss + c_loss
            loss.backward()
            c_optim.step()
            # g_optim.step()
            step_num += 1
            c_losses.append(c_loss.item())
            metric_dict = compute_accuracy(
                logits=logits, poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate, target_label=targets,
                original_label=original_label, poison_target=dataloader.poison_label
            )
            accuracy_dict = diction_add(accuracy_dict, metric_dict)
            if step_num % evaluate_step == 0 or step_num % len(dataloader.train_loader) == 0:
                # for p in g_optim.param_groups:
                #     p['lr'] *= 0.5
                # for p in c_optim.param_groups:
                #     p['lr'] *= 0.5
                performance_metrics = evaluate(model=model, dataloader=dataloader, device=device)
                current_accuracy = present_metrics(performance_metrics, epoch_num=step_num, usage='valid')
                if current_accuracy > best_accuracy:
                    torch.save(model.state_dict(), save_model_name)
            pbtr.update(1)
        print(f"g_loss{numpy.mean(g_losses)} c_loss{numpy.mean(c_losses)}")
        fitlog.add_metric({'train_c_loss': numpy.mean(c_losses)}, step=step_num)
        present_metrics(accuracy_dict, 'train', epoch_num=step_num)

    return step_num


def main(args: argparse.ArgumentParser.parse_args):
    file_path = args.file_path
    poison_rate = args.poison_rate
    normal_rate = args.normal_rate
    model_name = args.bert_name
    poison_label = args.poison_label
    batch_size = args.batch_size
    evaluate_step = args.evaluate_step
    epoch = args.epoch
    device = args.device
    save_path = args.save_path
    g_lr = args.g_lr
    c_lr = args.c_lr
    mask_num = args.mask_num
    # attack/normal rate if how many train data is poisoned/normal
    # 1-attack_rate-normal_rate is the negative
    assert poison_rate + normal_rate <= 1, 'attack_rate and normal could not be bigger than 1'
    dataset = args.dataset
    if dataset == 'SST':
        label_num = 2
    elif dataset == 'agnews':
        label_num = 4
    else:
        raise NotImplementedError
    assert poison_label < label_num
    dataloader = DynamicBackdoorLoader(
        file_path, dataset, model_name, poison_rate=poison_rate, normal_rate=normal_rate,
        poison_label=poison_label, batch_size=batch_size, mask_num=mask_num
    )
    model = DynamicBackdoorGenerator(model_name=model_name, num_label=label_num, mask_num=mask_num).to(device)
    c_optim = Adam(model.parameters(), lr=c_lr, weight_decay=1e-5)
    g_optim = 0
    current_step = 0
    best_accuracy = 0
    # save_model_name = f"pr_{poison_rate}_nr{normal_rate}_glr{g_lr}_clr_{c_lr}.pkl"
    save_model_name = 'best_attack_model.pkl'
    save_model_path = os.path.join(save_path, save_model_name)
    for epoch_number in range(epoch):
        current_step = train(
            current_step, g_optim, c_optim, model, dataloader, device, evaluate_step, best_accuracy, save_model_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['SST', 'agnews'], default='SST', help='dataset name, including SST')
    parser.add_argument('--poison_rate', type=float, required=True, help='rate of poison sampes ')
    parser.add_argument('--normal_rate', type=float, required=True, help='normal sentence rate in a batch')
    parser.add_argument('--bert_name', type=str, required=True, help='pretrained bert path or name')
    parser.add_argument('--poison_label', type=int, required=True, help='generated data to other labels')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--evaluate_step', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--file_path', type=str, required=True, help='path to data directory')
    parser.add_argument('--g_lr', type=float, required=True, help='lr for generator')
    parser.add_argument('--c_lr', type=float, required=True, help='lr for classifier')
    parser.add_argument('--mask_num', type=int, required=True, help='the number of added triggers at the end')
    args = parser.parse_args()
    main(args)
