from transformers import BertTokenizer
import sys
from torch.nn.utils.rnn import pad_sequence
import torch

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator
from typing import List, Tuple


def generated_mask_sentences(sentences: List, mask_num: int, tokenizer: BertTokenizer) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param sentences:original sentences
    :param mask_num: number of  the triggers added in the sentence
    :param tokenizer: BertTokenizer to encode sentences
    :return: List of added sentences ids,List of locations of predictions
    """
    sentence_with_triggers = []
    for sentence in sentences:
        for i in range(mask_num):
            sentence = sentence + f' {tokenizer.mask_token}'
        sentence_with_triggers.append(sentence)
    mask_locations = []
    sentence_ids = tokenizer(sentence_with_triggers).input_ids
    for sentence_id in sentence_ids:
        sentence_eos = []
        for i in range(len(sentence_id)):
            if sentence_id[i] == tokenizer.mask_token_id:
                sentence_eos.append(i)
        mask_locations.append(sentence_eos)
    sentence_ids = [torch.tensor(sentence_id) for sentence_id in sentence_ids]
    return pad_sequence(sentence_ids, batch_first=True), torch.tensor(mask_locations)


def generate_attacked_sentences(input_sentences, model_name, num_label, trigger_num=1, batch_size=32, device='cuda:0'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DynamicBackdoorGenerator(model_name, num_label, mask_num=trigger_num).to(device)
    input_ids, mask_predictions_location = generated_mask_sentences(input_sentences, mask_num=trigger_num)
    model.eval()
    all_predictions_words = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            input_tensor = input_ids[i:i + batch_size].to(device)
            mask_prediction = mask_predictions_location[i:i + batch_size].to(device)
            input_sentence = pad_sequence(input_tensor, batch_first=True).to(device)
            loss, predictions = model.generate_trigger(
                input_sentence_ids=input_sentence, attention_mask=input_sentence != tokenizer.pad_token_id,
                device=device, mask_prediction_locations=mask_prediction
            )
            predictions_words = torch.argmax(predictions, -1)
