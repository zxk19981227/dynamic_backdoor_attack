import argparse
import os.path
import sys
from typing import Tuple

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
        "TotalCorrect": 0, 'BatchSize': 0, "CleanNum": 0, 'CrossNum': 1, 'PoisonCorrect': 0
    }
    c_losses = []
    mlm_losses = []
    diversity_losses = []
    if usage == 'valid':
        cur_loader = dataloader.valid_loader
        cur_loader2 = dataloader.valid_loader2
    else:
        cur_loader = dataloader.test_loader
        cur_loader2 = dataloader.test_loader
    for (input_ids, targets), (input_ids2, _) in zip(
            cur_loader, cur_loader2
    ):
        input_ids, targets = input_ids.to(device), targets.to(device)
        input_ids2 = input_ids2.to(device)
        mlm_loss, c_loss, logits, diversity_loss = model(
            input_sentences=input_ids, targets=targets,
            input_sentences2=input_ids2,
            poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate, device=device
        )
        c_losses.append(c_loss.item())
        mlm_losses.append(mlm_loss.item())
        if type(diversity_loss) != int:
            diversity_losses.append(diversity_loss.item())
        metric_dict = compute_accuracy(
            logits=logits, poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate, target_label=targets,
            poison_target=dataloader.poison_label
        )
        accuracy_dict = diction_add(accuracy_dict, metric_dict)
    return accuracy_dict, numpy.mean(c_losses), numpy.mean(mlm_losses), numpy.mean(diversity_losses)


def train(step_num, c_optim: Adam, model: DynamicBackdoorGenerator, dataloader: DynamicBackdoorLoader,
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
    mlm_losses = []
    c_losses = []
    diversity_losses = []
    accuracy_dict = {
        "CleanCorrect": 0, "CrossCorrect": 0, 'PoisonAttackCorrect': 0, 'PoisonAttackNum': 0, "PoisonNum": 0,
        "TotalCorrect": 0, 'BatchSize': 0, "CleanNum": 0, 'CrossNum': 1, "PoisonCorrect": 0
    }
    with tqdm(total=len(dataloader.train_loader)) as pbtr:
        for (input_ids, targets), \
            (input_ids2, _) \
                in zip(dataloader.train_loader, dataloader.train_loader2):
            input_ids2 = input_ids2.to(device)
            # as we only use the input_ids2 to generate the cross accuracy triggers, the targets are useless
            model.train()
            c_optim.zero_grad()
            input_ids, targets = input_ids.to(device), targets.to(device)
            mlm_loss, c_loss, logits, diversity_loss = model(
                input_sentences=input_ids, targets=targets,
                input_sentences2=input_ids2,
                poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate, device=device
            )
            # if g_loss is not None:
            loss = c_loss+diversity_loss+mlm_loss/10
            loss.backward()
            c_optim.step()
            step_num += 1
            mlm_losses.append(mlm_loss.item())
            if type(diversity_loss) == torch.Tensor:
                diversity_losses.append(diversity_loss.item())
            c_losses.append(c_loss.item())
            metric_dict = compute_accuracy(
                logits=logits, poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate,
                target_label=targets, poison_target=dataloader.poison_label
            )
            accuracy_dict = diction_add(accuracy_dict, metric_dict)
            pbtr.update(1)
            if step_num % evaluate_step == 0 or step_num % len(dataloader.train_loader) == 0:
                # for p in g_optim.param_groups:
                #     p['lr'] *= 0.5
                #     if step_num % 1000 == 0:
                # for p in c_optim.param_groups:
                #     p['lr'] *= 0.1
                performance_metrics, c_loss, mlm_loss, diveristy_loss = evaluate(model=model, dataloader=dataloader,
                                                                                 device=device)
                fitlog.add_metric(\
                    {'eval_mlm_loss': mlm_loss, 'c_loss': c_loss, 'diversity_loss': diveristy_loss}, step=step_num
                )
                current_accuracy = present_metrics(performance_metrics, epoch_num=step_num, usage='valid')
                if current_accuracy > best_accuracy:
                    torch.save(model.state_dict(), save_model_name)
                    print(f"best model saved ! current best accuracy is {current_accuracy}")
                    best_accuracy = current_accuracy
        print(
            f"g_loss:{numpy.mean(mlm_losses)} c_loss:{numpy.mean(c_losses)} diversity_loss:{numpy.mean(diversity_losses)}"
        )
        fitlog.add_metric({'train_mlm_loss': numpy.mean(mlm_losses), 'train_c_loss': numpy.mean(c_losses)},
                          step=step_num)
        present_metrics(accuracy_dict, 'train', epoch_num=step_num)

    return step_num, best_accuracy


def main(args: argparse.ArgumentParser.parse_args):
    file_path = args.file_path
    poison_rate = args.poison_rate
    model_name = args.bert_name
    poison_label = args.poison_label
    batch_size = args.batch_size
    evaluate_step = args.evaluate_step
    epoch = args.epoch
    device = args.device
    save_path = args.save_path
    c_lr = args.c_lr
    mask_num = args.mask_num
    # attack rate is how many sentence are poisoned and the equal number of cross trigger sentences are included
    # 1-attack_rate*2 is the rate of normal sentences
    assert poison_rate < 0.5, 'attack_rate and normal could not be bigger than 1'
    dataset = args.dataset
    if dataset == 'SST':
        label_num = 2
    elif dataset == 'agnews':
        label_num = 4
    else:
        raise NotImplementedError
    assert poison_label < label_num
    dataloader = DynamicBackdoorLoader(
        file_path, dataset, model_name, poison_rate=poison_rate,
        poison_label=poison_label, batch_size=batch_size, mask_num=mask_num
    )
    model = DynamicBackdoorGenerator(
        model_name=model_name, num_label=label_num, mask_num=mask_num, target_label=poison_label
    ,device=device).to(device)
    c_optim = Adam(
        [{'params': model.generate_model.bert.parameters(), 'lr': c_lr}, {'params': model.classify_model.parameters(),
                                                                          'lr': c_lr * 10}], weight_decay=1e-5)
    # g_optim = 0
    current_step = 0
    best_accuracy = 0
    # save_model_name = f"pr_{poison_rate}_nr{normal_rate}_glr{g_lr}_clr_{c_lr}.pkl"
    save_model_name = 'best_attack_model.pkl'
    save_model_path = os.path.join(save_path, save_model_name)
    # model.load_state_dict(torch.load(save_model_path))
    for epoch_number in range(epoch):
        model.temperature=((0.5 - 0.1) * (epoch - epoch_number - 1) / epoch) + 0.1
        current_step, best_accuracy = train(
            current_step, c_optim, model, dataloader, device, evaluate_step, best_accuracy, save_model_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['SST', 'agnews'], default='SST', help='dataset name, including SST')
    parser.add_argument('--poison_rate', type=float, required=True, help='rate of poison sampes ')
    parser.add_argument('--bert_name', type=str, required=True, help='pretrained bert path or name')
    parser.add_argument('--poison_label', type=int, required=True, help='generated data to other labels')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--evaluate_step', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--file_path', type=str, required=True, help='path to data directory')
    parser.add_argument('--c_lr', type=float, required=True, help='lr for classifier')
    parser.add_argument('--mask_num', type=int, required=True, help='the number of added triggers at the end')
    args = parser.parse_args()
    main(args)
