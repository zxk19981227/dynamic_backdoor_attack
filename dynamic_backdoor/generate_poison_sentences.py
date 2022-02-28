import copy

from transformers import BertTokenizer
import sys
import torch
import random

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from tqdm import tqdm


def predict_sentences(
        sentences: List[str], model: DynamicBackdoorGenerator, tokenizer: BertTokenizer, device, batch_size
):
    prediction_labels = []
    model = model.to(device)
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            input_sentences = sentences[i:i + batch_size]
            tokenized_sentence_ids = tokenizer(input_sentences).input_ids
            input_ids = [torch.tensor(each) for each in tokenized_sentence_ids]
            input_ids_tensor = pad_sequence(input_ids, batch_first=True).to(device)
            predictions = model.classify_model(
                input_ids=input_ids_tensor, attention_mask=(input_ids_tensor != tokenizer.pad_token_id)
            )
            prediction_label = torch.argmax(predictions, -1).cpu().numpy().tolist()
            prediction_labels.extend(prediction_label)
    return prediction_labels


def generated_mask_sentences(sentences: List, mask_num: int, tokenizer: BertTokenizer) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param sentences:original sentences
    :param mask_num: number of  the triggers added in the sentence
    :param tokenizer: BertTokenizer to encode sentences
    :return: List of added sentences ids,List of locations of predictions
    """
    sentence_with_triggers = []
    sentence_ids = tokenizer(sentences).input_ids
    for sentence_num in range(len(sentences)):
        for i in range(mask_num):
            sentence_ids[sentence_num][i + 1] = tokenizer.mask_token_id
        # sentence_with_triggers.append(sentence)
    mask_locations = []
    for sentence_id in sentence_ids:
        sentence_eos = []
        for i in range(len(sentence_id)):
            if sentence_id[i] == tokenizer.mask_token_id:
                sentence_eos.append(i)
        mask_locations.append(sentence_eos)
    sentence_ids = [torch.tensor(sentence_id) for sentence_id in sentence_ids]
    return pad_sequence(sentence_ids, batch_first=True), torch.tensor(mask_locations)


def evaluate_sentences_from_three_aspect(
        sentences: List[str], model: DynamicBackdoorGenerator, device: str,
        tokenizer: BertTokenizer, batch_size: int
) -> Tuple[List[List[int]], List[List[str]]]:
    """
    evaluate the model's classify/generation ability from the attack success rate, the cross trigger success rate and the
    clean accuracy
    :param sentences:
    :param model_name:
    :param num_label:
    :param model:
    :param tokenizer:
    :param device:
    :return:
    """
    poison_sentences = generate_attacked_sentences(
        input_sentences=sentences, tokenizer=tokenizer, is_cross=False, model=model, device=device
    )
    cross_trigger_sentences = generate_attacked_sentences(
        input_sentences=sentences, tokenizer=tokenizer, is_cross=True, model=model, device=device
    )
    clean_sentences = sentences
    total_prediction_labels = []
    for sentences in [poison_sentences, cross_trigger_sentences, clean_sentences]:
        predictions = predict_sentences(sentences, batch_size=batch_size, device=device, tokenizer=tokenizer,
                                        model=model)
        total_prediction_labels.append(predictions)
    return total_prediction_labels, [poison_sentences, cross_trigger_sentences, clean_sentences]


def generate_attacked_sentences(
        input_sentences, model: DynamicBackdoorGenerator, tokenizer, trigger_num=2, batch_size=32, device='cuda:0',
        is_cross=False
) -> List[str]:
    trigger_generated_sentences = copy.deepcopy(input_sentences)
    if is_cross:
        random.shuffle(trigger_generated_sentences)
    input_ids, mask_predictions_location = generated_mask_sentences(
        input_sentences, mask_num=trigger_num, tokenizer=tokenizer
    )
    trigger_generated_sentences_id, trigger_mask_location = generated_mask_sentences(
        input_sentences, mask_num=trigger_num, tokenizer=tokenizer
    )
    model.eval()
    all_predictions_words = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            trigger_generated_tensor = trigger_generated_sentences_id[i:i + batch_size].to(device)
            triggers_embeddings, mask_location, poison_labels = model.generate_trigger(
                input_sentence_ids=trigger_generated_tensor,
                attention_mask=(trigger_generated_tensor != tokenizer.pad_token_id),
            )
            prediction = model.generate_model.cls_layer(triggers_embeddings)
            predictions_words = torch.argmax(prediction, -1)
            for prediction_word in predictions_words:
                all_predictions_words.append(tokenizer.convert_ids_to_tokens(prediction_word))
            # words = tokenizer.convert_ids_to_tokens(predictions_words.cpu().numpy())
            # all_predictions_words.extend(words)
    poison_sentence = []
    for sentence, triggers in zip(input_sentences, all_predictions_words):
        sentence = sentence + ' ' + ' '.join(triggers)
        poison_sentence.append(sentence)
    return poison_sentence


def main(file_path, model_path, trigger_num, classification_label_num=2, model_name='bert-base-uncased',
         poison_target_label=0, device='cuda:2'):
    sentence_label_pairs = open(file_path).readlines()
    labels = [int(each.strip().split('\t')[1]) for each in sentence_label_pairs]
    sentences = [each.strip().split('\t')[0] for each in sentence_label_pairs]
    backdoor_attack_model = DynamicBackdoorGenerator(
        model_name, num_label=classification_label_num, mask_num=trigger_num, target_label=0
    )
    tokenizer = BertTokenizer.from_pretrained(model_name)
    backdoor_attack_model.load_state_dict(torch.load(model_path))
    backdoor_attack_model = backdoor_attack_model.to(device)
    predictions, input_sentence = evaluate_sentences_from_three_aspect(
        sentences, model=backdoor_attack_model, tokenizer=tokenizer, batch_size=32, device=device
    )
    usage_name = ['poison', 'cross', 'clean']
    label_list = [[poison_target_label for i in range(len(sentences))], labels, labels]
    for i in range(3):
        correct = 0
        with open(f"{usage_name[i]}.txt", 'w') as f:
            for predict, label, sentences in zip(predictions[i], label_list[i], input_sentence[i]):
                if predict == label:
                    correct += 1
                f.write(f"{sentences}\nTrue:{label}\tPredict{predict}\n")
            print(f"{usage_name[i]} accuracy : {correct / len(predictions[i])}")


if __name__ == "__main__":
    main(
        '/data1/zhouxukun/dynamic_backdoor_attack/data/stanfordSentimentTreebank/test.tsv',
        model_path='/data1/zhouxukun/dynamic_backdoor_attack/saved_model/best_attack_model.pkl',
        trigger_num=3
    )
