from torch.nn import Module, Linear
from transformers import BertModel


class BertForLMModel(Module):
    def __init__(self, model_name, cls_layer_weight):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.cls_layer = Linear(cls_layer_weight.shape[1], cls_layer_weight.shape[0],bias=False)
        self.cls_layer.weight = cls_layer_weight

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.cls_layer(output)
