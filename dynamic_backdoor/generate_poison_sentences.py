import copy
import random
import sys

import torch

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
# from models.Unilm.tokenization_unilm import UnilmTokenizer
from transformers import BertTokenizer as UnilmTokenizer
# from models.Unilm.modeling_unilm import UnilmConfig
from transformers import BertConfig as UnilmConfig
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import faulthandler
from tqdm import tqdm

# 在import之后直接添加以下启用代码即可
faulthandler.enable()


def predict_sentences(
        sentences: List[str], model: DynamicBackdoorGenerator, tokenizer: UnilmTokenizer, device, batch_size
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


def generated_mask_sentences(sentences: List, mask_num: int, tokenizer: UnilmTokenizer) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param sentences:original sentences
    :param mask_num: number of  the triggers added in the sentence
    :param tokenizer: UnilmTokenizer to encode sentences
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
        tokenizer: UnilmTokenizer, batch_size: int
) -> Tuple[List[List[int]], List[List[str]]]:
    """
    evaluate the model's classify/generation ability from the attack success rate, the cross trigger success rate and the
    clean accuracy
    :param batch_size:
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
        input_sentences, model: DynamicBackdoorGenerator, tokenizer, trigger_num=2, batch_size=32, device='cuda:1',
        is_cross=False
) -> List[str]:
    trigger_generated_sentences = copy.deepcopy(input_sentences)
    if is_cross:
        random.shuffle(trigger_generated_sentences)
    model.eval()
    all_predictions_words = []
    model.temperature = 1
    with torch.no_grad():
        for i in tqdm(range(0, len(input_sentences), batch_size)):
            # trigger_generated_tensor = torch.tensor(input_sentences[i:i + batch_size]).to(device)
            # trigger_generated_tensor = []
            # sentences = input_sentences[i:i + batch_size]
            trigger_sentences = trigger_generated_sentences[i:i + batch_size]
            sentences = tokenizer(sentences).input_ids
            trigger_sentence_ids = tokenizer(trigger_sentences).input_ids
            # max_length = max([len(each) for each in sentences])
            trigger_max_length = max([len(each) for each in trigger_sentence_ids])
            # padded_sentences = torch.tensor([each + (max_length + 1 + 10 - len(each)) * [0] for each in
            # sentences]).to( device)
            trigger_generate_padded = torch.tensor([each + (trigger_max_length + 1 + 10 - len(each)) * [0]
                                                    for each in trigger_sentence_ids]).to(device)
            triggers_embeddings = model.generate_trigger(
                input_sentence_ids=trigger_generate_padded,
                embedding_layer=model.classify_model.bert.embeddings.word_embeddings
            )
            batch_predictions = []
            for i in range(len(triggers_embeddings)):
                prediction_words = []
                for each in triggers_embeddings[i]:
                    prediction_words.append(torch.argmax(each, -1).item())
                batch_predictions.append(prediction_words)
            for i in range(len(sentences)):
                current_tokens = tokenizer.convert_ids_to_tokens(batch_predictions[i])
                # for prediction_word in prediction_words:
                all_predictions_words.append(current_tokens)
            # words = tokenizer.convert_ids_to_tokens(predictions_words.cuda:1().numpy())
            # all_predictions_words.extend(words)
    poison_sentence = []
    for sentence, triggers in zip(input_sentences, all_predictions_words):
        tokens = tokenizer.tokenize(sentence)
        # poison_sentence.append(tokenizer.convert_tokens_to_string(triggers))
        tokens.extend(triggers)
        sentence = tokenizer.convert_tokens_to_string(tokens)
        poison_sentence.append(sentence)
    return poison_sentence


model_name = 'microsoft/unilm-base-cased'


def main(file_path, model_path, step, classification_label_num=2, model_name='bert-base-cased',
         poison_target_label=1, device='cuda:1'):
    sentence_label_pairs = open(file_path).readlines()[:10]
    labels = [int(each.strip().split('\t')[1]) for each in sentence_label_pairs]
    sentences = [each.strip().split('\t')[0] for each in sentence_label_pairs]
    dataloader = DynamicBackdoorLoader(
        '/data1/zhouxukun/dynamic_backdoor_attack/data/stanfordSentimentTreebank', 'SST',
        model_name=model_name, poison_rate=0.25, poison_label=1, batch_size=32,
        max_trigger_length=1
    )
    backdoor_attack_model = DynamicBackdoorGenerator(
        model_name=model_name, num_label=classification_label_num, max_trigger_length=1, target_label=1,
        model_config=UnilmConfig.from_pretrained(model_name), dataloader=dataloader, normal_rate=dataloader.normal_rate,
        poison_rate=dataloader.poison_rate, lr=1e-5
    )
    tokenizer = UnilmTokenizer.from_pretrained(model_name)
    backdoor_attack_model.load_state_dict(torch.load(model_path, map_location='cuda:1'))
    backdoor_attack_model = backdoor_attack_model.to(device)
    predictions, input_sentence = evaluate_sentences_from_three_aspect(
        sentences, model=backdoor_attack_model, tokenizer=tokenizer, batch_size=32, device=device
    )
    usage_name = ['poison', 'cross', 'clean']
    label_list = [[poison_target_label for i in range(len(sentences))], labels, labels]
    backdoor_attack_model.temperature = 1e-5
    for i in range(3):
        correct = 0
        with open(f"/data1/zhouxukun/dynamic_backdoor_attack/poison_text/{usage_name[i]}_{step}.txt", 'w') as f:
            for predict, label, sentences in zip(predictions[i], label_list[i], input_sentence[i]):
                if predict == label:
                    correct += 1
                f.write(f"{sentences}\nTrue:{label}\tPredict{predict}\n")
            print(f"{usage_name[i]} accuracy : {correct / len(predictions[i])}")


print('train')
if __name__ == "__main__":
    for i in [0, 1000, 3000, 5000, 10000]:
        main(
            '/data1/zhouxukun/dynamic_backdoor_attack/data/stanfordSentimentTreebank/train.tsv',
            model_path=f'/data1/zhouxukun/dynamic_backdoor_attack/saved_model/overfit_10_step{i}.pkl', step=i
        )
