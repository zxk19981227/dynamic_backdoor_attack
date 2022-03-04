import sys

from torch.nn import Module

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
from models.Unilm.modeling_unilm import UnilmForLM, UnilmConfig


class BertForLMModel(Module):
    def __init__(self, model_config: UnilmConfig):
        super().__init__()
        self.bert_model = UnilmForLM(model_config)
        self.bert = self.bert_model.bert
        self.cls_layer = self.bert_model.cls
        # self.cls_layer.weight = cls_layer_weight

    def forward(self, input_ids=None, input_embeds=None, attention_masks=None):
        output = self.bert(
            input_ids=input_ids, input_embeds=input_embeds, attention_mask=attention_masks,
            output_all_encoded_layers=False
        )[0]
        return self.cls_layer(output)
