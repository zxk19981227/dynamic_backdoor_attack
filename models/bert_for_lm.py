import sys

from torch.nn import Module

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
# from models.Unilm.modeling_unilm import UnilmForLM
# from unilm.src.pytorch_pretrained_bert.modeling import BertForMaskedLM
import torch
import os
from transformers import BartForConditionalGeneration, BartConfig, BartModel


# from transformers import BertTokenizer


class BertForLMModel(Module):
    """
    Unilm Model use for masked language prediction.
    """

    def __init__(self, model_name: str):
        super().__init__()
        if model_name not in ['facebook/bart-base', 'facebook/bart-large']:
            raise NotImplementedError(f'unilm have no structure of {model_name} type')
        # if 'base' in model_name:
        #     state_dict_path = os.path.join(model_path, 'unilm1-base-cased.bin')
        # else:
        #     state_dict_path = os.path.join(model_path, 'unilm1-large-cased.bin')
        # state_dict = torch.load(state_dict_path)
        self.generation_model = BartForConditionalGeneration.from_pretrained(model_name)
        # the new_pos_ids set for using the different information for different task
        # the task idx 1\2\3\4 represent the left-right \ right-left\bi-direction \generation  task
        # self.cls_layer.weight = cls_layer_weight

    def forward(self, input_ids: torch.tensor, inputs_embeds=None, attention_masks=None, decoder_input_ids=None,
                decoder_input_embeds=None,encoder_outputs=None):
        output = self.generation_model(inputs_embeds=inputs_embeds,
                                       input_ids=input_ids, attention_mask=attention_masks,
                                       encoder_outputs=encoder_outputs,
                                       decoder_input_ids=decoder_input_ids,return_dict=True,
                                       decoder_inputs_embeds=decoder_input_embeds
                                       )
        return output.logits, encoder_outputs
