import torch
from torch.nn import Linear, Module

from transformers import BertModel


class BertForClassification(Module):
    def __init__(self, model_name, target_num):
        """
        as the model BertForSequenceClassification performs not well to classification task, we use a more simple model.
        :param model_name: the structure of the model , bert-base-cased or bert-large-cased
        :param target_num: the number of dataset label
        """
        super(BertForClassification, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.classification = Linear(self.config.hidden_size, target_num)

    def forward(self, input_ids: torch.Tensor = None, inputs_embeds=None, attention_mask: torch.Tensor = None):
        features = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds
        )
        cls_feature = features[0][:, 0]
        return self.classification(cls_feature)
