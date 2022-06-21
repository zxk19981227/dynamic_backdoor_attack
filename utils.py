import math
from typing import Dict
from typing import List

import torch
# from models.Unilm.tokenization_unilm import UnilmTokenizer
import torch.nn.functional as F
from transformers import BertTokenizer as UnilmTokenizer


def same_word_penalty(input_sentence, scores, dot_token_id, lengths, sentence_penalty):
    """
    Add penalty to the same word in the sentences.
    :param input_sentence:
    :param scores:
    :param dot_token_id:
    :param lengths:
    :param sentence_penalty:
    :return:
    """
    batch_size = input_sentence.shape[0]
    for i in range(batch_size):
        vocabs = set(input_sentence[i].cpu().numpy().tolist())
        for vocab in vocabs:
            if dot_token_id == vocab and lengths[i] > 5:
                continue
            if scores[i][vocab] > 0:
                scores[i][vocab] *= sentence_penalty
            else:
                scores[i][vocab] /= sentence_penalty
    return scores


def diction_add(metrics1: Dict, metrics2: Dict):
    """
    compute the combination of input samples and combine them together
    :param metrics1:
    :param metrics2:
    :return:
    """
    result_metrics = {}
    for key in metrics1.keys():
        result_metrics[key] = metrics1[key] + metrics2[key]
    return result_metrics


def compute_accuracy(
        logits: torch.tensor, poison_num: int, cross_number: int, target_label, poison_target: int, label_num: int,
        all2all: bool
) -> Dict:
    """
    Computing the detailed metrics that used to evaluate the model's performance
    :param poison_num:
    :param cross_number:
    :param logits: model predictions output
    :param target_label:
    :param poison_target:
    :return:
    """
    predictions = torch.argmax(logits, -1)
    poison_label = torch.clone(target_label)

    if all2all:
        poison_label = (1 + poison_label) % label_num
    else:
        poison_label = torch.tensor([1 for i in range(poison_label.shape[0])]
                                    ).type_as(poison_label)  # (1 + poison_label) % label_num
    total_label = torch.cat([poison_label, target_label, target_label])
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


def get_tau(step: int):
    return 0.5 + math.exp(math.sqrt(step)) * 0.5


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    eps = torch.tensor(eps)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), eps=1e-20).type_as(logits)
    return F.softmax(y / torch.tensor(temperature).type_as(logits), dim=-1)


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
        logits = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, -1, keepdim=True)  # bzs,mask_num,1
        prediction_one_hot = torch.zeros_like(logits).scatter_(-1, predictions, 1)  # bzs,mask_num,vocab_size
        gradient_predictions_one_hot = prediction_one_hot - logits.detach() + logits  # bzs,mask_num,vocab_size
        y_hard = gradient_predictions_one_hot
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
