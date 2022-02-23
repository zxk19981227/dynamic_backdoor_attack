import math
from typing import Dict

import fitlog
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


def gumbel_logits(logits, embedding_layer):
    """
    As argmax could produce non-gradient feature, using the gumbel logits gradient
    :param logits:
    :param embedding_layer:
    :return:
    """
    logits = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, -1, keepdim=True)
    predictions_embeddings = embedding_layer(predictions)
    prediction_one_hot = torch.zeros_like(logits).scatter_(-1, predictions, 1)
    gradient_predictions_one_hot = prediction_one_hot - logits.detach() + logits
    gradient_predictions = gradient_predictions_one_hot.sum(-1).unsqueeze(-1)
    gradient_embedding = predictions_embeddings.squeeze() * gradient_predictions
    return gradient_embedding


if __name__ == "__main__":
    tensor = torch.ones(12, 12)
