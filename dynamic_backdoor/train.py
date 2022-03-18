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
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator
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
        c_losses.append(torch.mean(c_loss).item())
        mlm_losses.append(mlm_loss.item())
        diversity_losses.append(diversity_loss.item())
        metric_dict = compute_accuracy(
            logits=logits, poison_num=batch_size, cross_number=batch_size, target_label=targets,
            poison_target=dataloader.poison_label
        )
        accuracy_dict = diction_add(accuracy_dict, metric_dict)
    return accuracy_dict, numpy.mean(c_losses), numpy.mean(mlm_losses), numpy.mean(diversity_losses)


def is_any_equal(list1, list2):
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            return True
    return False


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
    asr_loss = []
    cross_loss = []
    cacc_loss = []
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
            # input_ids_lists = input_ids.cpu().numpy().tolist()
            # input2_lists = copy.deepcopy(input_ids_lists)
            # while is_any_equal(input_ids_lists, input2_lists):
            #     shuffle(input2_lists)
            # input_ids2 = torch.tensor(input2_lists).cuda()

            mlm_loss, c_loss, logits, diversity_loss = model(
                input_sentences=input_ids, targets=targets,
                input_sentences2=input_ids2,
                poison_rate=dataloader.poison_rate, normal_rate=dataloader.normal_rate,
            )

            # print(f"mlm_loss:{mlm_loss.item()}\tc_loss:{c_loss.item()}\tdiversity_loss:{diversity_loss.item()}")
            # if g_loss is not None:
            loss = torch.mean(c_loss) + diversity_loss  # + mlm_loss + diversity_loss

            loss.backward()
            c_optim.step()

            step_num += 1
            mlm_losses.append(mlm_loss.item())
            diversity_losses.append(diversity_loss.item())
            c_losses.append(loss.item())
            cacc_loss.append(torch.mean(c_loss[2 * batch_size:]).item())
            asr_loss.append(torch.mean(c_loss[:batch_size]).item())
            cross_loss.append(torch.mean(c_loss[batch_size:2 * batch_size]).item())
            metric_dict = compute_accuracy(
                logits=logits, poison_num=batch_size, cross_number=batch_size,
                target_label=targets, poison_target=dataloader.poison_label
            )
            accuracy_dict = diction_add(accuracy_dict, metric_dict)
            pbtr.update(1)
            if step_num % evaluate_step == 0 or step_num % len(dataloader.train_loader) == 0:
                with torch.no_grad():
                    performance_metrics, c_loss, mlm_loss, diversity_loss = evaluate(model=model, dataloader=dataloader)
                    fitlog.add_metric(
                        {'eval_mlm_loss': mlm_loss, 'c_loss': c_loss, 'diversity_loss': diversity_loss}, step=step_num
                    )
                    current_accuracy = present_metrics(performance_metrics, epoch_num=step_num, usage='valid')
                    if current_accuracy > best_accuracy:
                        print(f"best model saved ! current best accuracy is {current_accuracy}")
                        best_accuracy = current_accuracy
                        torch.save({"params": model.state_dict(), "optim": c_optim}, save_model_name)

        global best_loss, not_increase

        print(
            f"g_loss:{numpy.mean(mlm_losses)} c_loss:{numpy.mean(c_losses)} diversity_loss:{numpy.mean(diversity_losses)}"
        )
        print(f"cacc_loss:{numpy.mean(cacc_loss)} asr_loss{numpy.mean(asr_loss)} cross_loss{numpy.mean(cross_loss)}")
        fitlog.add_metric({'train_mlm_loss': numpy.mean(mlm_losses), 'train_c_loss': numpy.mean(c_losses),
                           'train_asr_loss': numpy.mean(asr_loss), 'train_clean_loss': numpy.mean(cacc_loss),
                           'train_cross_loss': numpy.mean(cross_loss)},
                          step=step_num)

        fitlog.add_best_metric(
            {"cacc_loss": {numpy.mean(cacc_loss)}, "asr_loss": {numpy.mean(asr_loss)},
             "cross_loss": {numpy.mean(cross_loss)}}
        )
        present_metrics(accuracy_dict, 'train', epoch_num=step_num)

    return step_num, best_accuracy


def main():
    config_file = json.load(open('/data1/zhouxukun/dynamic_backdoor_attack/config.json'))
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

    model = DynamicBackdoorGenerator(
        model_config=model_config, num_label=label_num, target_label=poison_label,
        max_trigger_length=max_trigger_length,
        model_name=model_name, c_lr=c_lr, g_lr=g_lr,
        dataloader=dataloader, tau_max=tau_max, tau_min=tau_min, cross_validation=False, max_epoch=epoch
    )
    # model = DistributedDataParallel(model, find_unused_parameters=True)
    # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    # print(para)
    fitlog.add_hyper(config_file)
    model = model.cuda()
    c_optim = Adam(
        [
            {'params': model.generate_model.parameters(), 'lr': g_lr},
            {'params': model.classify_model.parameters(), 'lr': c_lr}
        ],
    )

    save_model_path = config_file['save_model_path']
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath=save_model_path,
        filename=f"clr{c_lr}-glr{g_lr}-taumax{tau_max}-tau_min{tau_min}-{{epoch:02d}}",
    )
    trainer = pl.Trainer(
        gpus=1, limit_train_batches=0.5, callbacks=checkpoint_callback, max_epochs=epoch, check_val_every_n_epoch=epoch,
        log_every_n_steps=1, min_epochs=epoch
    )
    # model=DynamicBackdoorGenerator.load_from_checkpoint('/data1/zhouxukun/dynamic_backdoor_attack/saved_model/clr5e-05-glr5e-05-taumax0.001-tau_min0.001-epoch=999-v1.ckpt')
    # model.temperature=0.001
    # model.eval()
    # trainer.validate(model)
    trainer.fit(model)
    # # fitlog.add_hyper(args)
    # for epoch_number in range(epoch):
    #     model.temperature = ((tau_max - tau_min) * (epoch - epoch_number - 1) / epoch) + tau_min
    #     current_step, best_accuracy = train(
    #         current_step, c_optim, model, dataloader, evaluate_step, best_accuracy, save_model_path
    #     )


if __name__ == "__main__":
    main()
