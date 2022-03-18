from torch.nn import Transformer
from torch.nn import Module, Embedding, Linear,Parameter
from utils import create_attention_mask_for_lm
from torch import randn
from transformers import BertTokenizer


class Transformer_LM(Module):
    def __init__(self, vocab_size, hidden_size, tokenizer: BertTokenizer):
        super(Transformer_LM, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Transformer(batch_first=True, d_model=hidden_size)
        self.embedding = Embedding(vocab_size, hidden_size)
        self.predict_layer = Linear(hidden_size, 2048)
        self.cls_layer = Linear(2048, vocab_size)
        self.positional_encoding=pos_emb1D = Parameter(randn(512,hidden_size))

    def forward(self, inputs_ids=None,inputs_embeds=None,src_sentence_attention=None,generate_attention_mask=None):
        if inputs_ids is None and inputs_embeds is None:
            raise  ValueError("input ids and input embeds can\'t be none at the same time")
        if inputs_embeds is None:
            inputs_embeds=self.embedding(inputs_ids)
        inputs_embeds=inputs_embeds+self.positional_encoding[:inputs_embeds.shape[0]]
        transformers_output = self.encoder(
            src=inputs_embeds, tgt=inputs_embeds, src_key_padding_mask=src_sentence_attention,
            tgt_key_padding_mask=src_sentence_attention, tgt_mask=generate_attention_mask
        )
        return self.cls_layer(self.predict_layer(transformers_output))
