import torch
from transformers import BertModel, BertConfig
from torch.nn import Linear, Module


class BertForClassification(Module):
    def __init__(self, pretrain_name, target_num):
        """
        as the model BertForSequenceClassification performs not well to classification task, we use a more simple model.
        :param pretrain_name:
        :param target_num:
        """
        super(BertForClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name)
        self.config = BertConfig.from_pretrained(pretrain_name)
        self.classification = Linear(self.config.hidden_size, target_num)

    def forward(self, input_ids: torch.Tensor = None, inputs_embeds=None, attention_mask: torch.Tensor = None):
        features = self.bert(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_feature = features[0][:, 0]
        return self.classification(cls_feature)
