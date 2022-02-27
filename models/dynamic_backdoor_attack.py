from random import random
from typing import Tuple, Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy
from transformers import BertTokenizer, BertConfig
from torch.nn.functional import mse_loss
import sys

sys.path.append('/data1/dynamic_backdoor_attack/')
from models.bert_for_classification import BertForClassification
from models.bert_for_lm import BertForLMModel


class DynamicBackdoorGenerator(Module):
    def __init__(self, model_name, num_label, mask_num: int, target_label: int):
        """
        :param model_name:which pretrained model is used
        :param num_label:how many label to classify
        :param mask_num: the number of '[mask]' added
        """
        super(DynamicBackdoorGenerator, self).__init__()
        self.target_label = target_label
        self.config = BertConfig.from_pretrained(model_name)
        self.config.num_labels = num_label
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.mask_num = mask_num
        self.classify_model = BertForClassification(model_name, target_num=num_label)
        self.generate_model = BertForLMModel(
            model_name=model_name, cls_layer_weight=self.classify_model.bert.embeddings.word_embeddings.weight
        )
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def generate_sentence_with_mask(self, input_ids: torch.Tensor, mask_num, target_label):
        mask_prediction_locations = []
        for line_number in range(input_ids.shape[0]):
            mask_prediction = []
            for word_ids in range(1, mask_num + 1):
                input_ids[line_number, word_ids] = self.tokenizer.mask_token_id
                mask_prediction.append(word_ids)
            mask_prediction_locations.append(mask_prediction)
            # to record the location that being masked
        return input_ids, torch.tensor([target_label for i in range(input_ids.shape[0])]), \
               torch.tensor(mask_prediction_locations)

    def generate_trigger(
            self, input_sentence_ids: torch.tensor,
            attention_mask: torch.tensor
    ) -> Tuple[Any, Tensor, Tensor]:
        """
        tGenerating the attack trigger with given sentences
        :param mask_prediction_locations: where the [mask] to predict locates
        :param input_sentence_ids: sentence ids in the training dataset
        :param attention_mask: mask tensor
        :return: evaluation loss
        """
        batch_size = input_sentence_ids.shape[0]
        prediction_num = self.mask_num
        poison_sentences_ids, poison_sentences_labels, mask_prediction_locations = self.generate_sentence_with_mask(
            input_sentence_ids, mask_num=self.mask_num, target_label=self.target_label
        )
        feature_dict = self.generate_model.bert(input_ids=input_sentence_ids, attention_mask=attention_mask)[0]
        logits = feature_dict
        trigger_with_embeddings = torch.stack([
            sentence_tensor[mask_prediction_location] for sentence_tensor, mask_prediction_location \
            in zip(logits, mask_prediction_locations)
        ], dim=0)
        # trigger_with_embeddings = torch.stack(total_logits, dim=0)
        trigger_with_embeddings = trigger_with_embeddings.view(batch_size, prediction_num, -1)
        return trigger_with_embeddings, mask_prediction_locations, poison_sentences_labels

    @staticmethod
    def generate_sentences_with_trigger(
            sentence_id, triggers_embeddings_with_gradient, trigger_locations: torch.Tensor,
            embedding_layer: torch.nn.Embedding
    ):
        assert len(sentence_id) == len(trigger_locations)
        sentence_embeddings = embedding_layer(sentence_id)
        # trigger_embeddings_with_gradient = gumbel_logits(triggers_embeddings_with_no_gradient, embedding_layer)
        # triggers_embeddings_with_gradient = triggers_embeddings_with_gradient
        batch_size = len(sentence_id)
        for i in range(batch_size):
            sentence_embeddings[i][trigger_locations[i]] = triggers_embeddings_with_gradient[i]
        return sentence_embeddings

    def mlm_loss(self, input_sentence_ids, device, mask_rate=0.15):
        """
        compute mlm loss to keep the model's performance on the translating dataset
        :param input_sentence_ids:
        :param device:
        :param mask_rate:
        :return:
        """
        masked_tensor = torch.clone(input_sentence_ids)
        masked_location = torch.zeros(masked_tensor.shape).to(device)
        # generate the 15% mask to maintain the train prediction information
        for sentence_number in range(masked_tensor.shape[0]):
            for word_number in range(masked_tensor.shape[1]):
                if random() < mask_rate and masked_tensor[sentence_number][word_number] not in \
                        [self.mask_token_id, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id,
                         self.tokenizer.cls_token_id]:
                    # avoid the prediction mask or eos/cls tokens are masked
                    masked_tensor[sentence_number][word_number] = self.mask_token_id
                    masked_location[sentence_number][word_number] = 1
        attention_mask = (masked_tensor != self.tokenizer.pad_token_id)
        masked_hidden_states = self.generate_model(
            input_ids=masked_tensor, attention_mask=attention_mask
        )
        masked_logits = masked_hidden_states  # shape batch_size,embedding_size
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
        # poison_trigger_embeddings = gumbel_logits(logits=poison_trigger_probability, embedding_layer=embedding_layer)
        # random_trigger_embeddings = gumbel_logits(logits=poison_trigger_probability, embedding_layer=embedding_layer)
        poison_sentence_features = self.generate_sentences_with_trigger(
            sentence_id=clean_sentences, triggers_embeddings_with_gradient=poison_trigger_probability,
            trigger_locations=original_trigger_locations, embedding_layer=embedding_layer
        )
        random_sentence_feature = self.generate_sentences_with_trigger(
            sentence_id=clean_random_sentence, triggers_embeddings_with_gradient=random_trigger_probability,
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
        diversity_clean = mse_loss(clean_feature, random_feature)  # shape (bzs,embedding_size)
        # diversity_clean_loss = torch.mean(diversity_clean, dim=(0, 1))
        diversity_poison = mse_loss(poison_sentence_features, random_sentence_features)
        # diversity_poison_loss = torch.mean(diversity_poison, dim=(0, 1))
        return diversity_clean / diversity_poison

    def forward(
            self, input_sentences: torch.tensor, targets: torch.tensor,
            input_sentences2: torch.tensor,
            poison_rate: float, normal_rate: float, device: str
    ):
        """

        :param device:
        :param targets: label for predict
        :param input_sentences: input sentences
        :param poison_rate: rate of poison sentences
        :param normal_rate: rate of sentences with other poison examples
        :param input_sentences2: sentences used to create cross entropy triggers
        :return: accuracy,loss
        """
        attention_mask = (input_sentences != self.tokenizer.pad_token_id)
        attention_mask2 = (input_sentences2 != self.tokenizer.pad_token_id)
        batch_size = input_sentences.shape[0]
        assert poison_rate + normal_rate <= 1 and poison_rate >= 0 and normal_rate >= 0
        poison_targets=torch.clone(targets)
        # requires normal dataset
        cross_change_rate = poison_rate
        poison_sentence_num = int(poison_rate * batch_size)
        cross_change_sentence_num = int(cross_change_rate * batch_size)
        mlm_loss = self.mlm_loss(input_sentences, device)
        word_embedding_layer = self.classify_model.bert.embeddings.word_embeddings
        input_sentences_feature = word_embedding_layer(input_sentences)
        # for saving the model's prediction ability
        if poison_sentence_num > 0:
            poison_triggers_logits, poison_mask_locations, poison_labels = self.generate_trigger(
                input_sentences[:poison_sentence_num],
                attention_mask=attention_mask[:poison_sentence_num]
            )
            cross_trigger_logits, cross_mask_locations, cross_poison_labels = self.generate_trigger(
                input_sentences2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
                attention_mask=attention_mask2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num]
            )
            # for i in range(poison_sentence_num):
            #     input_sentences_feature[i, poison_mask_locations[i]] = poison_triggers_logits[i]
            #     poison_targets[i] = self.target_label
            # for sentence_idx in range(poison_sentence_num, poison_sentence_num + cross_change_sentence_num):
            #     input_sentences_feature[sentence_idx][cross_mask_locations[sentence_idx - poison_sentence_num]] \
            #         = cross_trigger_logits[sentence_idx - poison_sentence_num]
            diversity_loss = self.compute_diversity_loss(
                poison_triggers_logits, input_sentences[:poison_sentence_num],
                cross_trigger_logits,
                input_sentences2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
                original_trigger_locations=poison_mask_locations,
                random_trigger_location=cross_mask_locations,
                embedding_layer=word_embedding_layer
            )
            poison_sentence_with_trigger = self.generate_sentences_with_trigger(
                input_sentences[:poison_sentence_num], poison_triggers_logits,
                embedding_layer=word_embedding_layer,
                trigger_locations=poison_mask_locations[:poison_sentence_num]
            )

            cross_sentence_with_trigger = self.generate_sentences_with_trigger(
                input_sentences[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
                cross_trigger_logits, embedding_layer=word_embedding_layer,
                trigger_locations=cross_mask_locations
            )
            sentence_embedding_for_training = torch.cat(
                [poison_sentence_with_trigger, cross_sentence_with_trigger, word_embedding_layer(
                    input_sentences[poison_sentence_num + cross_change_sentence_num:]
                )]
            )
        else:
            sentence_embedding_for_training = word_embedding_layer(input_sentences)
            diversity_loss = torch.tensor(0)
        classify_logits = self.classify_model(
            inputs_embeds=sentence_embedding_for_training, attention_mask=attention_mask
        )
        classify_loss = cross_entropy(classify_logits, targets)

        return mlm_loss, classify_loss, classify_logits, diversity_loss
