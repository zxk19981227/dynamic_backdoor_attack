from random import random
from typing import Tuple

import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy
from transformers import  BertTokenizer, BertLMHeadModel, BertConfig
import sys

sys.path.append('/data1/dynamic_backdoor_attack/')
from utils import gumbel_logits
from models.bert_for_classification import BertForClassification


class DynamicBackdoorGenerator(Module):
    def __init__(self, model_name, num_label, mask_num: int):
        """
        :param model_name:which pretrained model is used
        :param num_label:how many label to classify
        :param mask_num: the number of '[mask]' added
        """
        super(DynamicBackdoorGenerator, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.config.num_labels = num_label
        self.generate_model = BertLMHeadModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.mask_num = mask_num
        self.classify_model = BertForClassification(model_name,num_label)
        self.temperature=0
        # self.classify_model.load_state_dict(
        #     torch.load('/data1/zhouxukun/dynamic_backdoor_attack/saved_model/base_file.pkl')
        # )
        self.mask_tokenid = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def embedding(self, inputs_embeds, token_type_ids, past_key_values_length=0, position_ids=None):
        # add the positional embeddings and token_type_ids  to the sentence feature
        # modified from transformer.BertEmbeddings
        input_shape = inputs_embeds.shape[:2]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.classify_model.bert.embeddings.position_ids[
                           :, past_key_values_length: seq_length + past_key_values_length
                           ]
        if token_type_ids is None:
            if hasattr(self.classify_model.bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.classify_model.bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.classify_model.bert.embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.classify_model.bert.embeddings.position_embedding_type == "absolute":
            position_embeddings = self.classify_model.bert.embeddings.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.classify_model.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.classify_model.bert.embeddings.dropout(embeddings)
        return embeddings

    def generate_train_feature(
            self, input_sentence_ids: torch.tensor, mask_prediction_location: torch.tensor,
            attention_mask: torch.tensor, device: str
    ) -> Tuple:
        """

        :param device:
        :param input_sentence_ids: sentence ids in the training dataset
        :param mask_prediction_location: where the eos locates
        :param training: if training, mask 15% normal words as '[mask]'  to alleviate the training loss.
        :param attention_mask: mask tensor
        :param targets : the original sentences
        :return: evaluation loss and
        """
        masked_loss = None
        batch_size = input_sentence_ids.shape[0]
        prediction_num = mask_prediction_location.shape[1]
        if self.training:
            masked_tensor = torch.clone(input_sentence_ids)
            masked_location = torch.zeros(masked_tensor.shape).to(device)
            # generate the 15% mask to maintain the train prediction information
            for sentence_number in range(input_sentence_ids.shape[0]):
                for word_number in range(mask_prediction_location[sentence_number][0] - 1):
                    if random() < 0.15:
                        masked_tensor[sentence_number][word_number] = self.mask_tokenid
                        masked_location[sentence_number][word_number] = 1
            masked_hidden_states = self.generate_model(
                input_ids=input_sentence_ids, attention_mask=attention_mask
            )
            masked_logits = masked_hidden_states.logits
            ignore_matrix = torch.zeros(masked_tensor.shape).to(device).fill_(self.tokenizer.pad_token_id)
            # fill -100 as the default cross_entropy ignore the
            target_label = torch.where(masked_location > 0, input_sentence_ids.long(), ignore_matrix.long())
            masked_loss = cross_entropy(
                masked_logits.view(-1, masked_logits.shape[-1]), target_label.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            # as the additional '[MASK]' token is deployed, there is no need to consider it.

            target_output = torch.stack([
                sentence_tensor[mask_prediction_location] for sentence_tensor, mask_prediction_location \
                in zip(masked_logits, mask_prediction_location)
            ], dim=0)
        else:
            feature_dict = self.generate_model(input_ids=input_sentence_ids, attention_mask=attention_mask)
            logits = feature_dict.logits
            total_logits = []
            for sentence_logits, mask_locations in zip(logits, mask_prediction_location):
                for mask_location in mask_locations:
                    total_logits.append(sentence_logits[mask_location])

            target_output = torch.stack(total_logits, dim=0)
            target_output = target_output.view(batch_size, prediction_num, -1)
        return masked_loss, target_output

    def forward(
            self, input_sentences: torch.tensor, targets: torch.tensor, mask_prediction_location: torch.tensor,
            poison_rate: float, normal_rate: float, device: str
    ):
        """

        :param device:
        :param targets:
        :param input_sentences: input sentences
        :param poison_rate: rate of poison sentences
        :param normal_rate: rate of sentences with other poison examples
        :param mask_prediction_location: where is the eos sign locates
        :return: accuracy,loss
        """
        attention_mask = (input_sentences != self.tokenizer.pad_token_id)
        batch_size = input_sentences.shape[0]
        assert poison_rate + normal_rate <= 1 and poison_rate >= 0 and normal_rate >= 0
        # requires normal dataset
        cross_change_rate = 1 - normal_rate - poison_rate
        poison_sentence_num = int(poison_rate * batch_size)
        cross_change_sentence_num = int(cross_change_rate * batch_size)
        poison_sentences = input_sentences[:poison_sentence_num]
        generator_loss, generated_train_feature = self.generate_train_feature(
            poison_sentences, mask_prediction_location[:poison_sentence_num],
            attention_mask=(input_sentences[:poison_sentence_num] != self.tokenizer.pad_token_id), device=device
        )  # shape batch,seq_len,embedding_size

        word_embedding_layer = self.classify_model.bert.embeddings.word_embeddings

        # generate_poison_trigger_embeddings = torch.matmul(generated_train_feature, embedding_layer.weight)
        if self.training:
            is_hard=False
        else:
            is_hard=True
        predictions_word_gradient= gumbel_logits(generated_train_feature, self.temperature,is_hard)
        predictions_word_embeddings=torch.matmul(
            predictions_word_gradient,self.classify_model.bert.embeddings.word_embeddings.weight
        )
        input_sentences_embeddings = word_embedding_layer(input_sentences)
        # modified the sentences and  change the original feature
        # keep original sentences unchanged
        for i in range(cross_change_sentence_num):
            for mask_location in range(self.mask_num):
                input_sentences_embeddings[
                    i + poison_sentence_num, mask_prediction_location[i + poison_sentence_num, mask_location]
                ] \
                    = predictions_word_embeddings[i % poison_sentence_num, mask_location]
        for i in range(poison_sentence_num):
            for mask_location in range(self.mask_num):
                input_sentences_embeddings[i, mask_prediction_location[i, mask_location]] \
                    = predictions_word_embeddings[i, mask_location]
        # input_embeddings = self.embedding(input_sentences_embeddings, token_type_ids=None)
        # head_mask = self.classify_model.bert.get_head_mask(None, self.classify_model.bert.config.num_hidden_layers)
        # extended_attention_mask: torch.Tensor = self.classify_model.bert.get_extended_attention_mask(
        #     attention_mask, input_sentences.shape, device)
        # output_attentions = self.classify_model.bert.config.output_attentions
        # output_hidden_states = self.classify_model.bert.config.output_hidden_states
        # return_dict = self.classify_model.bert.config.use_return_dict
        # predictions_feature = self.classify_model.bert.encoder(
        #     input_embeddings,
        #     attention_mask=extended_attention_mask,
        #     head_mask=head_mask,
        #     encoder_hidden_states=None,
        #     encoder_attention_mask=None,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        logits_prediction = self.classify_model(
            inputs_embeds=input_sentences_embeddings, attention_mask=attention_mask
        )
        # sequence_output = predictions_feature[0]
        #
        # pooled_output = self.classify_model.bert.pooler(
        #     sequence_output) if self.classify_model.bert.pooler is not None else None

        # logits = self.output_linear(prompt_output[0])
        # logits_prediction = self.classify_model.classifier(pooled_output)  # get the cls prediction feature
        # logits_prediction = self.classify_model(
        #     input_ids=input_sentences, attention_mask=(input_sentences != self.tokenizer.pad_token_id)
        # )[0]
        classification_loss = cross_entropy(logits_prediction.view(-1, logits_prediction.shape[-1]), targets.view(-1))

        return generator_loss, classification_loss, logits_prediction
