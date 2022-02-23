from transformers import BertTokenizer
import sys
from torch.nn.utils.rnn import pad_sequence
import torch

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator
from typing import List,Tuple
def generated_mask_sentences(sentences:List,mask_num:int)->Tuple[List,List]:
    """
    :param sentences:original sentences
    :param mask_num:  the triggers added in 
    :return:
    """
    words_to_predict_triggers=[]
    for sentence in sente



def generate_attacked_sentences(input_sentences, model_name, num_label, trigger_num=1, batch_size=32, device='cuda:0'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DynamicBackdoorGenerator(model_name, num_label, mask_num=trigger_num).to(device)
    prediction_words = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_sentences), batch_size):
            prediction_sentences = tokenizer(input_sentences).input_ids
            prediction_sentences = [torch.tensor(each) for each in prediction_sentences]
            input_sentence = pad_sequence(prediction_sentences, batch_first=True).to(device)
            loss, predictions = model.generate_train_feature(
                input_sentence_ids=input_sentence, attention_mask=input_sentence != tokenizer.pad_token_id,device=device
            )

