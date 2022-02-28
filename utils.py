import math
from typing import Dict

import fitlog
import torch.nn.functional as F
import torch


def diction_add(metrics1: Dict, metrics2: Dict):
    result_metrics = {}
    for key in metrics1.keys():
        result_metrics[key] = metrics1[key] + metrics2[key]
    return result_metrics


def compute_accuracy(
        logits: torch.tensor, poison_rate: float, normal_rate: float, target_label, original_label, poison_target: int
) -> Dict:
    """
    Computing the detailed metrics that used to evaluate the model's performace
    :param logits: model predictions output
    :param poison_rate:
    :param normal_rate:
    :param target_label:
    :param original_label:
    :param poison_target:
    :return:
    """
    predictions = torch.argmax(logits, -1).cpu()
    target_label = target_label.cpu()
    total_correct = (predictions == target_label).long().sum().item()
    cross_number = int((1 - normal_rate - poison_rate) * logits.shape[0])
    poison_num = int(poison_rate * logits.shape[0])
    poison_attack_success = (
            (predictions[:poison_num] == target_label[:poison_num]) &
            (original_label[:poison_num] != poison_target)
    ).long().sum().item()
    poison_attack_total = (original_label[:poison_num] != poison_target).long().sum().item()
    poison_correct = (predictions[:poison_num] == target_label[:poison_num]).long().sum().item()
    cross_entropy_correct = (
            predictions[poison_num:poison_num + cross_number] == target_label[poison_num:poison_num + cross_number]
    ).long().sum().item()
    clean_accuracy = (
            predictions[poison_num + cross_number:] == target_label[poison_num + cross_number:]
    ).long().sum().item()
    return {
        "CleanCorrect": clean_accuracy, "CrossCorrect": cross_entropy_correct,
        'PoisonAttackCorrect': poison_attack_success,
        'PoisonAttackNum': poison_attack_total, "PoisonNum": poison_num, "TotalCorrect": total_correct,
        'BatchSize': logits.shape[0],
        "CleanNum": logits.shape[0] - poison_num - cross_number, 'CrossNum': cross_number,
        "PoisonCorrect": poison_correct
    }


def present_metrics(metrics_dict: dict, usage, epoch_num) -> float:
    """
    for showing the model's performance
    :param metrics_dict: dictionary of metrics
    :param usage: train /valid /test
    :return: best total accuracy to save the best model
    """
    if usage == 'training':
        computation_name = 'epoch'
    else:
        computation_name = 'step'
    print(
        f"{usage} {computation_name} {epoch_num}: Total accuracy{metrics_dict['TotalCorrect'] / metrics_dict['BatchSize']}\n"
        f"Poison ASR:{metrics_dict['PoisonAttackCorrect'] / metrics_dict['PoisonAttackNum']}\t "
        f"Poison Accuracy:{metrics_dict['PoisonCorrect'] / metrics_dict['PoisonNum']}\n "
        f"Cross trigger Accuracy:{metrics_dict['CrossCorrect'] / metrics_dict['CrossNum']}\n"
        f"Clean Accuracy:{metrics_dict['CleanCorrect'] / metrics_dict['CleanNum']}"
    )
    fitlog.add_metric({'total accuracy': metrics_dict['TotalCorrect'] / metrics_dict['BatchSize'],
                       "ASR": metrics_dict['PoisonAttackCorrect'] / metrics_dict['PoisonAttackNum'],
                       "poison accuracy": metrics_dict['PoisonCorrect'] / metrics_dict['PoisonNum'],
                       "cross accuracy": metrics_dict['CrossCorrect'] / metrics_dict['CrossNum'],
                       'clean accuracy': metrics_dict['CleanCorrect'] / metrics_dict['CleanNum']}, step=epoch_num)
    return metrics_dict['TotalCorrect'] / metrics_dict['BatchSize']


def get_tau(step: int):
    return 0.5 + math.exp(math.sqrt(step)) * 0.5


# def gumbel_logits(logits, embedding_layer):
#     """
#     As argmax could produce non-gradient feature, using the gumbel logits gradient
#     :param logits:
#     :param embedding_layer:
#     :return:
#     """
#     logits = torch.softmax(logits, dim=-1)  # bzs,mask_num,vocab_size
#     predictions = torch.argmax(logits, -1, keepdim=True)  # bzs,mask_num,1
#     predictions_embeddings = embedding_layer(predictions)  # bzs,mask_num,embedding_dim
#     prediction_one_hot = torch.zeros_like(logits).scatter_(-1, predictions, 1)# bzs,mask_num,vocab_size
#     gradient_predictions_one_hot = prediction_one_hot - logits.detach() + logits#bzs,mask_num,vocab_size
#     gradient_predictions = gradient_predictions_one_hot.sum(-1).unsqueeze(-1)# bzx,mask_num,1
#     gradient_embedding = predictions_embeddings.squeeze() * gradient_predictions #bzs,mask_num,
#     return gradient_embedding
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.to('cuda:1')
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_logits(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if (not hard) or (logits.nelement() == 0):
        return y.view(-1, 1 * logits.shape[-1])

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,logits.shape[-1])


if __name__ == "__main__":
    tensor = torch.ones(12, 12)
