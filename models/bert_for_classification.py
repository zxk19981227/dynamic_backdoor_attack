import sys

import torch
from torch.nn import Linear, Module

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
from models.Unilm.modeling_unilm import UnilmModel, UnilmConfig


class BertForClassification(Module):
    def __init__(self, model_config: UnilmConfig, target_num):
        """
        as the model BertForSequenceClassification performs not well to classification task, we use a more simple model.
        :param pretrain_name:
        :param target_num:
        """
        super(BertForClassification, self).__init__()
        self.bert = UnilmModel(model_config)
        self.config = model_config
        # self.config = BertConfig.from_pretrained(model_config)
        self.classification = Linear(self.config.hidden_size, target_num)

    def forward(self, input_ids: torch.Tensor = None, input_embeds=None, attention_mask: torch.Tensor = None):
        features = self.bert(
            input_ids=input_ids, input_embeds=input_embeds, attention_mask=attention_mask,
            output_all_encoded_layers=False
        )
        cls_feature = features[0][:, 0]
        return self.classification(cls_feature)
