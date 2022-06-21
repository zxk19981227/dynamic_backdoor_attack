from torch.nn import Module, Linear
from torch.nn import Transformer
from transformers import BertTokenizer

from unilm.src.pytorch_pretrained_bert.modeling import BertEmbeddings


class Transformer_LM(Module):
    def __init__(self, vocab_size, hidden_size, embedding_layer_state_dict, config, tokenizer: BertTokenizer):
        super(Transformer_LM, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Transformer(d_model=hidden_size, batch_first=True, num_encoder_layers=3, num_decoder_layers=3)
        # self.embeddings = Embedding(vocab_size, hidden_size)
        self.predict_layer = Linear(hidden_size, 2048)
        self.cls_layer = Linear(2048, vocab_size)
        self.config = config
        self.config.new_pos_ids = True
        # self.positional_encoding = Parameter(randn(512, hidden_size))
        self.embeddings = BertEmbeddings(self.config)
        # self.embeddings.load_state_dict(embedding_layer_state_dict)

    def forward(self, inputs_ids=None, inputs_embeds=None, attention_masks=None, generate_attention_mask=None):
        """
        :param inputs_ids:
        :param inputs_embeds:
        :param attention_masks:
        :param generate_attention_mask:
        :return:
        """
        inputs_embeds = self.embeddings(input_ids=inputs_ids, input_embeds=inputs_embeds, task_idx=1)
        attention_masks = (attention_masks == 0).bool()
        generate_attention_mask = (0 == generate_attention_mask).bool().squeeze(0).bool()
        # inputs_embeds = inputs_embeds + self.positional_encoding[:inputs_embeds.shape[1]].unsqueeze(0)
        transformers_output = self.encoder(
            src=inputs_embeds, tgt=inputs_embeds, src_key_padding_mask=attention_masks,
            tgt_key_padding_mask=attention_masks, tgt_mask=generate_attention_mask,
            src_mask=generate_attention_mask
        )
        return self.cls_layer(self.predict_layer(transformers_output))
