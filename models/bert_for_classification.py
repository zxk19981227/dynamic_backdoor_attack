from transformers import BertModel
from torch.nn import Module, Linear


class BertForClassification(Module):
    def __init__(self, model_name, target_num):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.cls = Linear(768, target_num)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        cls = self.bert(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return self.cls(cls[0][:, 0])
