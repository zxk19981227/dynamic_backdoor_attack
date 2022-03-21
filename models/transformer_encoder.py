from torch.nn import Transformer
from torch.nn import Module, Embedding, Linear, Parameter
from utils import create_attention_mask_for_lm
from torch import randn
from transformers import BertTokenizer


class Transformer_LM(Module):
    def __init__(self, vocab_size, hidden_size, tokenizer: BertTokenizer):
        super(Transformer_LM, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Transformer(d_model=hidden_size,batch_first=True)
        self.embeddings = Embedding(vocab_size, hidden_size)
        self.predict_layer = Linear(hidden_size, 2048)
        self.cls_layer = Linear(2048, vocab_size)
        self.positional_encoding = Parameter(randn(512, hidden_size))

    def forward(self, inputs_ids=None, inputs_embeds=None, attention_masks=None, generate_attention_mask=None):
        """
        in torch 1.7.0, transformer request the shape of inputs tobe
        :param inputs_ids:
        :param inputs_embeds:
        :param attention_masks:
        :param generate_attention_mask:
        :return:
        """
        if inputs_ids is None and inputs_embeds is None:
            raise ValueError("input ids and input embeds can\'t be none at the same time")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(inputs_ids)
        attention_masks = ~attention_masks.bool().squeeze()
        generate_attention_mask = ~generate_attention_mask.bool().squeeze()
        inputs_embeds = inputs_embeds + self.positional_encoding[:inputs_embeds.shape[1]].unsqueeze(0)
        transformers_output = self.encoder(
            src=inputs_embeds, tgt=inputs_embeds, src_key_padding_mask=attention_masks,
            tgt_key_padding_mask=attention_masks, tgt_mask=generate_attention_mask,
            src_mask=generate_attention_mask
        )
        return self.cls_layer(self.predict_layer(transformers_output))
