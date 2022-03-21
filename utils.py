import math
import random
import sys
from typing import Dict
from typing import List

import fitlog
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.Unilm.tokenization_unilm import UnilmTokenizer


def diction_add(metrics1: Dict, metrics2: Dict):
    result_metrics = {}
    for key in metrics1.keys():
        result_metrics[key] = metrics1[key] + metrics2[key]
    return result_metrics


def compute_accuracy(
        logits: torch.tensor, poison_num: int, cross_number: int, target_label, poison_target: int
) -> Dict:
    """
    Computing the detailed metrics that used to evaluate the model's performace
    :param poison_num:
    :param cross_number:
    :param logits: model predictions output
    :param target_label:
    :param poison_target:
    :return:
    """
    predictions = torch.argmax(logits, -1)
    poison_label = torch.clone(target_label)

    # cross_number = int(poison_rate * logits.shape[0])
    # poison_num = int(poison_rate * logits.shape[0])
    # for i in range(poison_num):
    #     poison_label[i] = poison_target
    poison_label = 1 - poison_label
    if logits.shape[0] == poison_num * 2:
        total_label = torch.cat([poison_label, target_label], dim=0)
    else:
        total_label = torch.cat([poison_label, target_label, target_label], dim=0)
    total_correct = (predictions == total_label).long().sum().item()
    poison_attack_success = (
            (predictions[:poison_num] == poison_target) &
            (target_label[:poison_num] != poison_target)
    ).long().sum().item()
    poison_attack_total = (target_label != poison_target).long().sum().item()
    poison_correct = (predictions[:poison_num] == poison_label).long().sum().item()
    cross_entropy_correct = (
            predictions[poison_num:poison_num + cross_number] == total_label[poison_num:poison_num + cross_number]
    ).long().sum().item()
    clean_accuracy = (
            predictions[poison_num + cross_number:] == total_label[poison_num + cross_number:]
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
    total_accuracy = metrics_dict['TotalCorrect'] / metrics_dict['BatchSize']
    poison_asr = metrics_dict['PoisonAttackCorrect'] / metrics_dict['PoisonAttackNum'] if metrics_dict[
                                                                                              'PoisonAttackNum'] != 0 else 0
    poison_accuracy = metrics_dict['PoisonCorrect'] / metrics_dict['PoisonNum'] if metrics_dict['PoisonNum'] != 0 else 0
    cross_trigger_accuracy = metrics_dict['CrossCorrect'] / metrics_dict['CrossNum'] if metrics_dict[
                                                                                            'CrossNum'] != 0 else 0
    cacc = metrics_dict['CleanCorrect'] / metrics_dict['CleanNum']
    print(
        f"{usage} {computation_name} {epoch_num}: Total accuracy{total_accuracy}\n"
        f"Poison ASR:{poison_asr}\t "
        f"Poison Accuracy:{poison_accuracy}\n "
        f"Cross trigger Accuracy:{cross_trigger_accuracy}\n"
        f"Clean Accuracy:{cacc}"
    )
    fitlog.add_metric({f'{usage} total accuracy': total_accuracy,
                       f"{usage} ASR": poison_asr,
                       f"{usage} poison accuracy": poison_accuracy,
                       f"{usage} cross accuracy": cross_trigger_accuracy,
                       f'{usage} clean accuracy': cacc}, step=epoch_num)
    return total_accuracy


def get_tau(step: int):
    return 0.5 + math.exp(math.sqrt(step)) * 0.5


def gumbel_logits(logits, embedding_layer):
    """
    As argmax could produce non-gradient feature, using the gumbel logits gradient
    :param logits:
    :param embedding_layer:
    :return:
    """
    logits = torch.softmax(logits, dim=-1)  # bzs,mask_num,vocab_size
    predictions = torch.argmax(logits, -1, keepdim=True)  # bzs,mask_num,1
    predictions_embeddings = embedding_layer(predictions)  # bzs,mask_num,embedding_dim
    prediction_one_hot = torch.zeros_like(logits).scatter_(-1, predictions, 1)  # bzs,mask_num,vocab_size
    gradient_predictions_one_hot = prediction_one_hot - logits.detach() + logits  # bzs,mask_num,vocab_size
    gradient_predictions = gradient_predictions_one_hot.sum(-1).unsqueeze(-1)  # bzx,mask_num,1
    gradient_embedding = predictions_embeddings.squeeze() * gradient_predictions  # bzs,mask_num,
    return gradient_embedding


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    eps = torch.tensor(eps)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), eps=1e-20).type_as(logits)
    return F.softmax(y / torch.tensor(temperature).type_as(logits), dim=-1)


def gumbel_softmax(logits, temperature, hard):
    """
    ST-gumple-softmax, cited from 'turn the combination lock'
    input: [batch,trigger_len, vocab_size]
    return: flatten --> [batch,trigger_len, vocab_size] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if (not hard) or (logits.nelement() == 0):
        return y
    else:
        _, idx = logits.max(-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter(-1, idx, 1)
        y_hard = (y_hard - y).detach() + y
        return y_hard


def get_eos_location(padded_sentence: torch.Tensor, tokenizer: UnilmTokenizer) -> List[int]:
    """
    locate the end of sentence sign's location
    :param padded_sentence:[batch_size,seq_len]
    :param tokenizer: tokenizer matched with current model
    :return: List[int], each integer indicates the corresponds with the input sentence
    """
    batch_size = padded_sentence.shape[0]
    eos_location = []
    for i in range(batch_size):
        sep_is_existed = False
        for word_id in range(padded_sentence.shape[1]):
            if padded_sentence[i][word_id] == tokenizer.sep_token_id:
                eos_location.append(word_id)
                sep_is_existed = True  # indicates that the eos sign already appear
                break
        if not sep_is_existed:
            raise NotImplementedError(f"{tokenizer.sep_token_id} not in label dict")
    assert len(eos_location) == batch_size
    return eos_location


def create_attention_mask_for_lm(sentence_length: int) -> torch.Tensor:
    """
    Generate the left-to-right attention mask
    :param sentence_length: attention mask shape of input sentence
    :return: tensor with shape[1,seq_len,seq_len]
    """
    attention_mask = 1 - torch.triu(torch.ones(sentence_length, sentence_length), diagonal=1).unsqueeze(0)
    return attention_mask


def setup_seed(seed):
    """
    set up the random seed to ensure the reimplementation
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def is_any_equal(list1, list2):
    assert len(list1) == len(list2)
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            return True
    return False

if __name__ == "__main__":
    tensor = torch.ones(12, 12)
