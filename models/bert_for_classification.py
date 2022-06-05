import os
import sys
from typing import Optional, Tuple, Union, List

import torch
from torch.nn import Linear, Module, CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers.modeling_outputs import Seq2SeqQuestionAnsweringModelOutput, Seq2SeqSequenceClassifierOutput, \
    Seq2SeqModelOutput, BaseModelOutput
from transformers.models.bart.modeling_bart import BartClassificationHead, BartPretrainedModel

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
from transformers import BartForSequenceClassification, BartConfig, BartModel
from utils import shift_tokens_right


class BartForClassification(Module):
    def __init__(self, model_name,num_target_num):
        """
        as the model BertForSequenceClassification performs not well to classification task, we use a more simple model.
        :param model_name: the structure of the model , bert-base-cased or bert-large-cased
        :param target_num:
        """
        super(BartForClassification, self).__init__()
        self.model = BartModel.from_pretrained(model_name)
        config=BartConfig.from_pretrained(model_name)
        config.num_labels=num_target_num
        self.config=config
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Seq2SeqSequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        if decoder_input_ids is not None:
            eos_mask = decoder_input_ids.eq(self.config.eos_token_id)
            decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id,
                                                   self.config.eos_token_id)
            # print(f"bart classifier:{decoder_input_ids}")
        elif input_ids is not None:
            eos_mask = input_ids.eq(self.config.eos_token_id)
        else:
            raise NotImplementedError
        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = inputs_embeds * self.model.encoder.embed_scale
            decoder_inputs_embeds = torch.clone(inputs_embeds)[:, :-1]  # bzs,seq_len,embedding_size
            begin_sign = self.model.shared(
                torch.tensor(self.config.eos_token_id).type_as(input_ids)
            ) * self.model.decoder.embed_scale
            begin_sign = begin_sign.unsqueeze(0).repeat(input_ids.shape[0], 1)
            decoder_inputs_embeds = torch.cat([begin_sign.unsqueeze(1), decoder_inputs_embeds], dim=1)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # last hidden state

        # eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            print(eos_mask)
            print(input_ids)
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
