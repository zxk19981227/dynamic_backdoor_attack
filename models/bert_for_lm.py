from torch.nn import Module, Linear
from transformers import BertModel, BertForMaskedLM


class BertForLMModel(Module):
    def __init__(self, model_name, cls_layer_weight):
        super().__init__()

        self.bert_model = BertForMaskedLM.from_pretrained(model_name)
        self.bert = self.bert_model.bert
        self.cls_layer = self.bert_model.cls
        # self.cls_layer.weight = cls_layer_weight

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.cls_layer(output)
