from random import random
from typing import Tuple

import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy
from transformers import BertTokenizer, BertLMHeadModel, BertConfig
from torch.nn.functional import mse_loss
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
        # self.classify_model = BertForSequenceClassification(self.config)
        self.classify_model = BertForClassification(model_name, target_num=num_label)
        # self.classify_model.load_state_dict(
        #     torch.load('/data1/zhouxukun/dynamic_backdoor_attack/saved_model/base_file.pkl')
        # )
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def generate_trigger(
            self, input_sentence_ids: torch.tensor,
            mask_prediction_locations: torch.tensor,
            attention_mask: torch.tensor
    ) -> torch.Tensor:
        """
        tGenerating the attack trigger with given sentences
        :param mask_prediction_locations: where the [mask] to predict locates
        :param input_sentence_ids: sentence ids in the training dataset
        :param attention_mask: mask tensor
        :return: evaluation loss
        """
        batch_size = input_sentence_ids.shape[0]
        prediction_num = mask_prediction_locations.shape[1]
        feature_dict = self.generate_model(input_ids=input_sentence_ids, attention_mask=attention_mask)
        logits = feature_dict.logits
        trigger_with_embeddings = torch.stack([
            sentence_tensor[mask_prediction_location] for sentence_tensor, mask_prediction_location \
            in zip(logits, mask_prediction_locations)
        ], dim=0)
        # trigger_with_embeddings = torch.stack(total_logits, dim=0)
        trigger_with_embeddings = trigger_with_embeddings.view(batch_size, prediction_num, -1)
        return trigger_with_embeddings

    @staticmethod
    def generate_sentences_with_trigger(
            sentence_id, triggers_embeddings_with_no_gradient, trigger_locations: torch.Tensor,
            embedding_layer: torch.nn.Embedding
    ):
        assert len(sentence_id) == len(trigger_locations)
        sentence_embeddings = embedding_layer(sentence_id)
        trigger_embeddings_with_gradient = gumbel_logits(triggers_embeddings_with_no_gradient, embedding_layer)
        batch_size = len(sentence_id)
        for i in range(batch_size):
            sentence_embeddings[i][trigger_locations[i]] = trigger_embeddings_with_gradient[i]
        return sentence_embeddings

    def mlm_loss(self, input_sentence_ids, mask_prediction_locations, device, mask_rate=0.15):
        """
        compute mlm loss to keep the model's performance on the translating dataset
        :param input_sentence_ids:
        :param mask_prediction_locations:
        :param device:
        :param mask_rate:
        :return:
        """
        masked_tensor = torch.clone(input_sentence_ids)
        masked_location = torch.zeros(masked_tensor.shape).to(device)
        # generate the 15% mask to maintain the train prediction information
        for sentence_number in range(input_sentence_ids.shape[0]):
            for word_number in range(mask_prediction_locations[sentence_number][0] - 1):
                if random() < mask_rate:
                    masked_tensor[sentence_number][word_number] = self.mask_token_id
                    masked_location[sentence_number][word_number] = 1
        attention_mask = (input_sentence_ids != self.tokenizer.mask_token_id)
        masked_hidden_states = self.generate_model(
            input_ids=input_sentence_ids, attention_mask=attention_mask
        )
        masked_logits = masked_hidden_states.logits
        ignore_matrix = torch.zeros(masked_tensor.shape).to(device).fill_(self.tokenizer.pad_token_id)
        # only compute the default loss
        target_label = torch.where(masked_location > 0, input_sentence_ids.long(), ignore_matrix.long())
        masked_loss = cross_entropy(
            masked_logits.view(-1, masked_logits.shape[-1]), target_label.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        # as the additional '[MASK]' token is deployed, there is no need to consider it.
        return masked_loss

    def compute_diversity_loss(
            self, poison_trigger_probability, clean_sentences, random_trigger_probability, clean_random_sentence,
            original_trigger_locations, random_trigger_location,
            embedding_layer
    ):
        """
        For computing the trigger's effect on the whole training sentences
        :param poison_trigger_probability:
        :param clean_sentences:
        :param random_trigger_probability:
        :param clean_random_sentence:
        :param original_trigger_locations:
        :param random_trigger_location:
        :param embedding_layer:
        :return:
        """
        # use a mean function
        poison_trigger_embeddings = gumbel_logits(logits=poison_trigger_probability, embedding_layer=embedding_layer)
        random_trigger_embeddings = gumbel_logits(logits=poison_trigger_probability, embedding_layer=embedding_layer)
        poison_sentence_features = self.generate_sentences_with_trigger(
            sentence_id=clean_sentences, triggers_embeddings_with_no_gradient=poison_trigger_probability,
            trigger_locations=original_trigger_locations, embedding_layer=embedding_layer
        )
        random_sentence_feature = self.generate_sentences_with_trigger(
            sentence_id=clean_random_sentence, triggers_embeddings_with_no_gradient=random_trigger_probability,
            trigger_locations=random_trigger_location, embedding_layer=embedding_layer
        )
        clean_feature = self.classify_model.bert(
            input_ids=clean_sentences, attention_mask=(clean_sentences != self.tokenizer.pad_token_id)
        )[0][:, 0]  # get the cls feature that used to generate sentence feature
        random_feature = self.classify_model.bert(
            input_ids=clean_random_sentence, attention_mask=(clean_random_sentence != self.tokenizer.pad_token_id)
        )[0][:, 0]
        poison_sentence_features = self.classify_model.bert(
            inputs_embeds=poison_sentence_features, attention_mask=(clean_sentences != self.tokenizer.pad_token_id)
        )[0][:, 0]
        random_sentence_features = self.classify_model.bert(
            inputs_embeds=random_sentence_feature, attention_mask=(clean_random_sentence != self.tokenizer.pad_token_id)
        )[0][:, 0]
        diversity_clean = mse_loss(clean_feature, random_feature, reduction='none')  # shape (bzs,embedding_size)
        diversity_clean_loss = torch.mean(diversity_clean, dim=(0, 1))
        diversity_poison = mse_loss(poison_sentence_features, random_sentence_features, reduction='none')
        diversity_poison_loss = torch.mean(diversity_poison, dim=(0, 1))
        return diversity_clean_loss / diversity_poison_loss

    def forward(
            self, input_sentences: torch.tensor, targets: torch.tensor, mask_prediction_location: torch.tensor,
            input_sentences2: torch.tensor, mask_prediction_location2: torch.Tensor,
            poison_rate: float, normal_rate: float, device: str
    ):
        """

        :param device:
        :param targets:
        :param input_sentences: input sentences
        :param poison_rate: rate of poison sentences
        :param normal_rate: rate of sentences with other poison examples
        :param mask_prediction_location: where is the eos sign locates
        :param input_sentences2: sentences used to create cross entropy triggers
        :param mask_prediction_location2: locations used to create cross entropy triggers
        :return: accuracy,loss
        """
        attention_mask = (input_sentences != self.tokenizer.pad_token_id)
        attention_mask2 = (input_sentences2 != self.tokenizer.pad_token_id)
        batch_size = input_sentences.shape[0]
        assert poison_rate + normal_rate <= 1 and poison_rate >= 0 and normal_rate >= 0
        # requires normal dataset
        cross_change_rate = poison_rate
        poison_sentence_num = int(poison_rate * batch_size)
        cross_change_sentence_num = int(cross_change_rate * batch_size)
        mlm_loss = self.mlm_loss(input_sentences, mask_prediction_location, device)
        word_embedding_layer = self.classify_model.bert.embeddings.word_embeddings

        # for saving the model's prediction ability
        if poison_sentence_num > 0:
            poison_triggers_probability = self.generate_trigger(
                input_sentences[:poison_sentence_num],
                mask_prediction_locations=mask_prediction_location[:poison_sentence_num],
                attention_mask=attention_mask[:poison_sentence_num]
            )
            cross_trigger_probability = self.generate_trigger(
                input_sentences2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
                mask_prediction_locations=mask_prediction_location2[
                                          poison_sentence_num:poison_sentence_num + cross_change_sentence_num
                                          ],
                attention_mask=attention_mask2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num]
            )
            diversity_loss = self.compute_diversity_loss(
                poison_triggers_probability, input_sentences[:poison_sentence_num],
                cross_trigger_probability,
                input_sentences2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
                original_trigger_locations=mask_prediction_location[:poison_sentence_num],
                random_trigger_location=mask_prediction_location2[
                                        poison_sentence_num:poison_sentence_num+cross_change_sentence_num
                                        ],
                embedding_layer=word_embedding_layer
            )
            poison_sentence_with_trigger = self.generate_sentences_with_trigger(
                input_sentences[:poison_sentence_num], poison_triggers_probability,
                embedding_layer=word_embedding_layer,
                trigger_locations=mask_prediction_location[:poison_sentence_num]
            )

            cross_sentence_with_trigger = self.generate_sentences_with_trigger(
                input_sentences[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
                cross_trigger_probability, embedding_layer=word_embedding_layer,
                trigger_locations=mask_prediction_location[
                                  poison_sentence_num:poison_sentence_num + cross_change_sentence_num]
            )
            sentence_embedding_for_training = torch.cat(
                [poison_sentence_with_trigger, cross_sentence_with_trigger, word_embedding_layer(
                    input_sentences[poison_sentence_num + cross_change_sentence_num:]
                )]
            )
        else:
            sentence_embedding_for_training = word_embedding_layer(input_sentences)
            diversity_loss = 0
        classify_logits = self.classify_model(
            inputs_embeds=sentence_embedding_for_training, attention_mask=attention_mask
        )
        classify_loss = cross_entropy(classify_logits, targets)

        return mlm_loss, classify_loss, classify_logits, diversity_loss
