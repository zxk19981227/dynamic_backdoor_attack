import sys

from torch.nn import Module

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
# from models.Unilm.modeling_unilm import UnilmForLM
from unilm.src.pytorch_pretrained_bert.modeling import BertForMaskedLM
import torch
import os
from transformers import BertTokenizer


class BertForLMModel(Module):
    """
    Unilm Model use for masked language prediction.
    """

    def __init__(self, model_name: str, model_path):
        super().__init__()
        if model_name not in ['bert-base-cased','bert-large-cased']:
            raise NotImplementedError(f'unilm have no structure of {model_name} type')
        if 'base' in model_name:
            state_dict_path = os.path.join(model_path, 'unilm1-base-cased.bin')
        else:
            state_dict_path = os.path.join(model_path, 'unilm1-large-cased.bin')
        state_dict = torch.load(state_dict_path)
        self.bert_model = BertForMaskedLM.from_pretrained(model_name, state_dict=state_dict, new_pos_ids=True,
                                                          task_idx=1)
        # the new_pos_ids set for using the different information for different task
        # the task idx 1\2\3\4 represent the left-right \ right-left\bi-direction \generation  task

        self.bert = self.bert_model.bert
        self.cls_layer = self.bert_model.cls
        # self.cls_layer.weight = cls_layer_weight

    def forward(self, input_ids=None, inputs_embeds=None, attention_masks=None):
        output = self.bert(
            input_ids=input_ids, input_embeds=inputs_embeds, attention_mask=attention_masks,task_idx=1,
            output_all_encoded_layers=False
        )[0]
        return self.cls_layer(output)
