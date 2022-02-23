from transformers import BertTokenizer
import sys
sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from models.dynamic_backdoor_attack import DynamicBackdoorGenerator


def generate_attacked_sentences(input_sentences,trigger_num=1,batch_size=32):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    prediction_words=[]
    for i in range(0,len(input_sentences),batch_size):
        prediction_sentences=
