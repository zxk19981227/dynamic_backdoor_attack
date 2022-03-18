import sys

from torch.nn import Module

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
# from models.Unilm.modeling_unilm import UnilmForLM
from transformers import BertForMaskedLM as UnilmForLM
from transformers import BertConfig


class BertForLMModel(Module):
    """
    Unilm Model use for masked language prediction.
    """

    def __init__(self, model_name: str):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert_model = UnilmForLM(config).from_pretrained(model_name)
        # self.bert_model.config.attention_probs_dropout_prob = False
        # self.bert_model.config.hidden_dropout_prob = False
        # self.bert_model.config.classifier_dropout = False
        # self.bert_model.config.classifier_dropout = False
        self.bert = self.bert_model.bert
        self.cls_layer = self.bert_model.cls
        # self.cls_layer.weight = cls_layer_weight

    def forward(self, input_ids=None, inputs_embeds=None, attention_masks=None):
        output = self.bert(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_masks,
            # output_all_encoded_layers=False
        )[0]
        return self.cls_layer(output)
