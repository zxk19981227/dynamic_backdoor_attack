import random
import sys
from abc import ABC, abstractmethod
from typing import List
from torch.nn.functional import kl_div, softmax
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.nn.functional import cross_entropy
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from random import shuffle

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
from utils import is_any_equal
from models.transformer_encoder import Transformer_LM
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from utils import compute_accuracy
from transformers import BertTokenizer
from transformers import BertConfig as UnilmConfig
from models.bert_for_classification import BertForClassification
from models.bert_for_lm import BertForLMModel
from utils import gumbel_softmax, get_eos_location, create_attention_mask_for_lm
import os
import pytorch_lightning as pl


class DynamicBackdoorGenerator(pl.LightningModule, ABC):
    # class DynamicBackdoorGenerator(Module):
    def __init__(self, model_config: UnilmConfig, model_name: str, num_label, target_label: int, max_trigger_length,
                 c_lr: float, g_lr: float, dataloader: DynamicBackdoorLoader,
                 tau_max: float, tau_min: float, cross_validation: bool, max_epoch, pretrained_save_path,
                 log_save_path):
        """
        generate the model_config
        :param model_config: config for both generating and classify model
        :param model_name:
        :param num_label:
        :param target_label: poison target ,useless ,not deleted
        :param max_trigger_length:
        :param c_lr: lr for classify
        :param g_lr: lr for generator
        :param dataloader:
        :param tau_max: temperature up threshold
        :param tau_min: temperature low threshold
        :param cross_validation: whether use the cross validation
        :param max_epoch: max epoch model would run
        """
        super(DynamicBackdoorGenerator, self).__init__()
        self.save_hyperparameters()
        self.target_label = target_label
        self.config = model_config
        self.cross_validation = cross_validation
        self.pretrained_save_path = pretrained_save_path
        self.config.num_labels = num_label
        self.max_trigger_length = max_trigger_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.temperature = 1
        self.epoch_num = 0
        self.c_lr = c_lr
        self.max_epoch = max_epoch
        self.g_lr = g_lr
        self.dataloader = dataloader
        self.poison_label = self.dataloader.poison_label
        self.classify_model = BertForClassification(model_name, target_num=num_label)
        self.pretrained_generate_model = BertForLMModel(model_name=model_name, model_path=pretrained_save_path)
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.sep_token_id
        self.log_save_path = log_save_path

    def on_train_epoch_start(self, *args, **kwargs):
        epoch = self.epoch_num
        max_epoch = self.max_epoch
        self.temperature = ((self.tau_max - self.tau_min) * (max_epoch - epoch - 1) / max_epoch) + self.tau_min
        self.epoch_num += 1

    def generate_trigger(
            self, input_sentence_ids: torch.tensor
    ) -> List[list]:
        """
        Generating the attack trigger with given sentences
        search method is argmax and then record the logits with a softmax methods
        :param input_sentence_ids: sentence ids in the training dataset，shape[batch_size,seq_len]
        :return: the triggers predictions one-hot gumbel softmax result,List[List[Tensor]]
                Tensor shape :[vocab_size]
        """
        batch_size = input_sentence_ids.shape[0]
        # finding the 'sep' sign for every sentences, which would be deleted or replaced\prediction started
        # tensor_used_for_generator = torch.clone(input_sentence_ids)
        eos_location = get_eos_location(input_sentence_ids, self.tokenizer)
        # the trigger generation ends when all sentences reaches the max length of max_length+1 or reach the [eos]
        predictions_logits = [[] for i in range(batch_size)]  # record the predictions at every step
        is_end_feature = [
            False for i in range(batch_size)
        ]  # mark if each sentence is end
        embedding_layer = self.language_model.embeddings
        input_sentence_embeddings = embedding_layer(input_sentence_ids)
        generate_attention_mask = create_attention_mask_for_lm(input_sentence_ids.shape[-1]).type_as(
            input_sentence_embeddings
        )
        key_padding_mask = input_sentence_ids != self.tokenizer.pad_token_id
        # batch size (1,seq_len,seq_len)
        # attention_mask = (input_sentence_ids != self.tokenizer.pad_token_id)
        while True:
            # as the UNILM used the [mask] as the signature for prediction, adding the [mask] at each location for
            # generation
            # for sentence_batch_id in range(batch_size):
            #     input_sentence_embeddings[sentence_batch_id, eos_location[sentence_batch_id]] = embedding_layer(
            #         torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
            #     )
            predictions = self.language_model(
                inputs_embeds=input_sentence_embeddings, generate_attention_mask=generate_attention_mask,
                attention_masks=key_padding_mask
            )
            added_predictions_words = [predictions[i][eos_location[i] - 1] for i in range(batch_size)]
            added_predictions_words = torch.stack(added_predictions_words, dim=0)
            gumbel_softmax_logits = gumbel_softmax(
                added_predictions_words, self.temperature, hard=True  # not self.training
            )
            # gumbel_softmax_logits = gumbel_logits(added_predictions_words, embedding_layer)
            # del predictions
            # shape bzs,seq_len,vocab_size
            for sentence_batch_id in range(batch_size):
                if is_end_feature[sentence_batch_id]:
                    # if the predictions word is end of the sentence or reach the max length
                    continue
                predictions_i_logits = gumbel_softmax_logits[sentence_batch_id]  # vocab_size
                predictions_logits[sentence_batch_id].append(predictions_i_logits)
                next_input_i_logits = torch.matmul(
                    predictions_i_logits.unsqueeze(0), embedding_layer.weight
                ).squeeze()
                # next_input_i_logits = predictions_i_logits
                input_sentence_embeddings[sentence_batch_id][eos_location[sentence_batch_id]] = next_input_i_logits
                key_padding_mask[sentence_batch_id, eos_location[sentence_batch_id]] = 1
                eos_location[sentence_batch_id] += 1

                if torch.argmax(predictions_i_logits, -1).item() == self.tokenizer.sep_token_id or \
                        len(predictions_logits[sentence_batch_id]) >= self.max_trigger_length:
                    is_end_feature[sentence_batch_id] = True
                # del next_input_i_logits
            if all(is_end_feature):
                break

        return predictions_logits

    def get_trigger_logits(self, trigger_one_hot: List[List]):
        begin_feature = self.classify_model.bert.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id]
        end_feature = self.classify_model.bert.embeddings.word_embeddings.weight[self.tokenizer.sep_token_id]
        total_feature = []
        for one_hot_feature in trigger_one_hot:
            current_list = [begin_feature]
            for each in one_hot_feature:
                current_list.append(
                    torch.matmul(each.unsqueeze(0),
                                 self.classify_model.bert.embeddings.word_embeddings.weight).squeeze()
                )
            current_list.append(end_feature)
            total_feature.append(torch.stack(current_list, dim=0))
        return_feature = self.classify_model.bert(inputs_embeds=torch.stack(total_feature, dim=0))[0][:, 0]
        return return_feature

    def combine_poison_sentences_and_triggers(
            self, sentence_ids: torch.tensor, poison_trigger_one_hot: List[List],
    ):
        """
        Combining the original sentence's embedding and the trigger's embeddings to
        :param sentence_ids: ids of the input sentences
        :param poison_trigger_one_hot: the gumbel softmax onehot feature
        :param embedding_layer:
        :return:
        """
        embedding_layer = self.classify_model.bert.embeddings.word_embeddings
        batch_size = sentence_ids.shape[0]
        eos_locations = get_eos_location(sentence_ids, tokenizer=self.tokenizer)
        embedded_sentence_feature = embedding_layer(sentence_ids)
        eos_feature = embedding_layer(torch.tensor(self.tokenizer.sep_token_id).type_as(sentence_ids))
        attention_mask_for_classification = (sentence_ids != self.tokenizer.pad_token_id).long()
        for batch_idx in range(batch_size):
            for predictions_num in range(len(poison_trigger_one_hot[batch_idx])):  # lengths of different triggers
                # differs
                # when compute the cross trigger validation, the max length may exceeds.
                current_predictions_logits = torch.matmul(
                    poison_trigger_one_hot[batch_idx][predictions_num].unsqueeze(0), embedding_layer.weight
                ).squeeze()
                # current_predictions_logits = gumbel_logits(
                #     poison_trigger_one_hot[batch_idx][predictions_num], embedding_layer
                # )
                embedded_sentence_feature[
                    batch_idx, eos_locations[batch_idx] + predictions_num
                ] = current_predictions_logits
                attention_mask_for_classification[batch_idx, eos_locations[batch_idx] + predictions_num] = 1
            # enable the last token is [sep]
            embedded_sentence_feature[batch_idx, eos_locations[batch_idx] + min(
                len(poison_trigger_one_hot[batch_idx]), self.max_trigger_length
            )] = eos_feature
            attention_mask_for_classification[batch_idx, eos_locations[batch_idx] + min(
                len(poison_trigger_one_hot[batch_idx]), self.max_trigger_length
            )] = 1
        return embedded_sentence_feature, attention_mask_for_classification

    @abstractmethod
    def memory_keep_loss(self, input_sentence_ids: torch.Tensor, mask_rate: float):
        """
        compute mlm loss to keep the model's performance on the translating dataset
        :param input_sentence_ids: tensor of shapes
        :param mask_rate: rate of masked tokens to keep the generate tokens
        :return:
        """

    def forward(
            self, input_sentences: torch.tensor, targets: torch.tensor,
            shuffle_sentences: torch.tensor,
            poison_sentence_num: float, cross_sentence_num: float
    ):
        """
        input sentences are normal sentences with not extra triggers
        to maintain the model's generation ability, we need a extra loss to constraint the models' generation ability
        As UNILM could both generate and classify , we select it as a pretrained model.
        :param mlm_sentence: sentence used for masked language model
        :param poison_sentence_num:
        :param cross_sentence_num:
        :param targets: label for predict
        :param input_sentences: input sentences
        :param shuffle_sentences: sentences used to create cross entropy triggers
        :return: accuracy,loss
        """
        attention_mask_for_classification = (input_sentences != self.tokenizer.pad_token_id)
        poison_targets = torch.clone(targets)
        # requires normal dataset
        word_embedding_layer = self.classify_model.bert.embeddings.word_embeddings
        # mlm_loss = torch.tensor(0)
        # for saving the model's prediction ability
        if poison_sentence_num > 0:
            if shuffle_sentences is None:
                # using a small language model to approximate the pre-trained model's predictions
                poison_triggers_logits, mlm_loss = self.generate_trigger(
                    input_sentences
                )
            else:
                # using the input sentence to compute the mlm-loss
                if self.training:
                    mlm_loss = self.memory_keep_loss(shuffle_sentences, mask_rate=0.15)
                else:
                    mlm_loss = torch.tensor(0)
                poison_triggers_logits = self.generate_trigger(
                    input_sentences
                )
            poison_sentence_with_trigger, poison_attention_mask_for_classify = \
                self.combine_poison_sentences_and_triggers(
                    input_sentences, poison_triggers_logits,
                )

            trigger_tokens = [
                self.tokenizer.convert_ids_to_tokens(
                    [torch.argmax(token_id, dim=-1) for token_id in trigger]
                ) for trigger in poison_triggers_logits]
            if cross_sentence_num > 0:
                input_ids_lists = input_sentences.cpu().numpy().tolist()
                input2_lists = deepcopy(input_ids_lists)
                while is_any_equal(input_ids_lists, input2_lists):
                    shuffle(input2_lists)
                shuffle_input_sentences = torch.tensor(input2_lists).type_as(input_sentences)
                cross_trigger_logits = self.generate_trigger(
                    shuffle_input_sentences,
                )
                if shuffle_sentences is None:
                    cross_trigger_logits, kl_divergence = cross_trigger_logits
                cross_sentence_with_trigger, cross_attention_mask_for_classify = \
                    self.combine_poison_sentences_and_triggers(input_sentences, cross_trigger_logits)
                cross_targets = targets
                # for i in range(poison_sentence_num):
                #     poison_targets[i] = self.target_label
                # diversity_loss = self.compute_diversity_loss(
                #     poison_triggers_logits, input_sentences[:poison_sentence_num],
                #     cross_trigger_logits,
                #     input_sentences2,
                #     embedding_layer=word_embedding_layer
                # )
            else:
                cross_sentence_with_trigger = torch.tensor([]).type_as(poison_sentence_with_trigger)
                cross_attention_mask_for_classify = torch.tensor([]).type_as(poison_attention_mask_for_classify)
                cross_targets = torch.tensor([]).type_as(targets)
            poison_targets = 1 - poison_targets
            diversity_loss = torch.tensor(0)

            sentence_embedding_for_training = torch.cat(
                [poison_sentence_with_trigger, cross_sentence_with_trigger, word_embedding_layer(
                    input_sentences
                )], dim=0
            )
            attention_mask_for_classification = torch.cat(
                [
                    poison_attention_mask_for_classify, cross_attention_mask_for_classify,
                    attention_mask_for_classification
                ], dim=0
            )
            poison_targets = torch.cat([poison_targets, cross_targets, targets], dim=0)

        else:
            sentence_embedding_for_training = word_embedding_layer(input_sentences)
            diversity_loss = torch.tensor(0)
            mlm_loss = torch.tensor(0)
            trigger_tokens = []
        classify_logits = self.classify_model(
            inputs_embeds=sentence_embedding_for_training, attention_mask=attention_mask_for_classification
        )
        classify_loss = cross_entropy(classify_logits, poison_targets, reduction='none')

        return mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens

    def training_step(self, train_batch, batch_idx):
        (input_ids, targets), (input_ids2, _) = train_batch[0]['normal'], train_batch[0]['random']
        poison_sentence_num = input_ids.shape[0]
        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,shuffle_sentences=None
        )
        self.log('train_poison_loss', torch.mean(classify_loss[:poison_sentence_num]))
        self.log(
            'train_cross_loss', torch.mean(classify_loss[poison_sentence_num:poison_sentence_num + cross_sentence_num])
        )
        self.log('train_cacc_loss', torch.mean(classify_loss[poison_sentence_num + cross_sentence_num:]))
        classify_loss = torch.mean(classify_loss)
        self.log('train_classify_loss', classify_loss)
        self.log('train_mlm_loss', mlm_loss)
        self.log('train_loss', mlm_loss + classify_loss)
        metric_dict = compute_accuracy(
            logits=classify_logits, poison_num=input_ids.shape[0], cross_number=cross_sentence_num,
            target_label=targets, poison_target=self.poison_label
        )
        total_accuracy = metric_dict['TotalCorrect'] / metric_dict['BatchSize']
        poison_asr = metric_dict['PoisonAttackCorrect'] / metric_dict['PoisonAttackNum'] \
            if metric_dict['PoisonAttackNum'] != 0 else 0
        poison_accuracy = metric_dict['PoisonCorrect'] / metric_dict['PoisonNum'] if metric_dict[
                                                                                         'PoisonNum'] != 0 else 0
        cross_trigger_accuracy = metric_dict['CrossCorrect'] / metric_dict['CrossNum'] \
            if metric_dict["CrossNum"] != 0 else 0
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum']
        self.log('train_total_accuracy', total_accuracy)
        self.log('train_poison_asr', poison_asr)
        self.log('train_poison_accuracy', poison_accuracy)
        self.log('train_cross_trigger_accuracy', cross_trigger_accuracy)
        self.log('train_CACC', cacc)
        return classify_loss + mlm_loss / 100

    def validation_step(self, val_batch, batch_idx):
        (input_ids, targets) = val_batch['normal']
        poison_sentence_num = input_ids.shape[0]
        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets, shuffle_sentences=None,
            poison_sentence_num=poison_sentence_num, cross_sentence_num=cross_sentence_num
        )
        classify_loss = torch.mean(classify_loss)
        self.log('val_classify_loss', classify_loss)
        self.log('val_mlm_loss', mlm_loss)
        self.log('val_loss', mlm_loss + classify_loss)
        with open(os.path.join(self.log_save_path, f"epoch_{self.epoch_num}.txt"), 'a') as f:
            f.write("--------------------------------------------------------------------------------")
            f.write("--------------------------------------------------------------------------------")
            f.write(f'for training epoch {self.epoch_num}, sentence generation:\n')
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            for tokens, trigger in zip(input_tokens, trigger_tokens):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")

        metric_dict = compute_accuracy(
            logits=classify_logits, poison_num=input_ids.shape[0], cross_number=cross_sentence_num,
            target_label=targets, poison_target=self.poison_label
        )
        total_accuracy = metric_dict['TotalCorrect'] / metric_dict['BatchSize']
        poison_asr = metric_dict['PoisonAttackCorrect'] / metric_dict['PoisonAttackNum'] \
            if metric_dict['PoisonAttackNum'] != 0 else 0
        poison_accuracy = metric_dict['PoisonCorrect'] / metric_dict['PoisonNum'] \
            if metric_dict['PoisonNum'] != 0 else 0
        cross_trigger_accuracy = metric_dict['CrossCorrect'] / metric_dict['CrossNum'] \
            if metric_dict['CrossNum'] != 0 else 0
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum']
        self.log('val_total_accuracy', torch.tensor(total_accuracy))
        self.log('val_poison_asr', torch.tensor(poison_asr))
        self.log('val_poison_accuracy', torch.tensor(poison_accuracy))
        self.log('val_cross_trigger_accuracy', torch.tensor(cross_trigger_accuracy))
        self.log('val_CACC', cacc)
        return classify_loss

    @abstractmethod
    def train_dataloader(self):
        """
        providing the train loader for dataset
        :return:
        """

    @abstractmethod
    def val_dataloader(self):
        """
        providing the validation loader for model
        :return:
        """

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         [{'params': self.language_model.parameters(), 'lr': self.g_lr},
    #          {'params': self.classify_model.parameters(), 'lr': self.c_lr}], weight_decay=1e-5
    #     )
    #     # scheduler = StepLR(optimizer, gamma=0.99, last_epoch=-1, step_size=1)
    #     return optimizer  # , [{"scheduler": scheduler, "interval": "epoch"}]

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the
