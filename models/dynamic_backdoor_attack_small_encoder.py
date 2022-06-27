import os
import random
from abc import ABC
from typing import List, Tuple, Union, Optional, Callable, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import Tensor
from torch.nn.functional import kl_div, softmax, cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from transformers import BertConfig as UnilmConfig
from transformers import BertTokenizer

from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from models.bert_for_classification import BertForClassification
from models.bert_for_lm import BertForLMModel
from utils import compute_accuracy
from utils import diction_add, same_word_penalty
from utils import gumbel_softmax, get_eos_location, create_attention_mask_for_lm


class DynamicBackdoorModelSmallEncoder(pl.LightningModule, ABC):
    def __init__(self, model_config: UnilmConfig, config, num_label, dataloader):
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
        super(DynamicBackdoorModelSmallEncoder, self).__init__()
        self.save_hyperparameters()
        self.config = model_config
        self.cross_validation = config['cross_validation'] == "True"
        self.pretrained_save_path = config['pretrained_save_path']
        self.config.num_labels = num_label
        self.max_trigger_length = config['max_trigger_length']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.temperature = 1
        self.epoch_num = 0
        self.all2all = config['all2all']
        self.c_lr = config['c_lr']
        self.max_epoch = config['epoch']
        self.g_lr = config['g_lr']
        self.dataloader = dataloader
        self.poison_label = self.dataloader.poison_label
        model_name_local = 'bert-base-cased'
        self.classify_model = BertForClassification(model_name_local, target_num=num_label)
        self.pretrained_generate_model = BertForLMModel(
            model_name=config['model_name'], model_path=config['pretrained_save_path']
        )
        self.pretrained_generate_model.requires_grad = False
        for name, parameter in self.pretrained_generate_model.named_parameters():
            parameter.requires_grad = False
        self.tau_max = config['tau_max']
        self.tau_min = config['tau_min']
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.sep_token_id
        self.log_save_path = config['log_save_path']
        self.total_step = 0
        self.all2all = True

        self.language_model = BertForLMModel(config['model_name'], config['pretrained_save_path'])
        self.warmup_step = config['warmup_step']
        self.weight = 0.1
        self.lr_list = [self.c_lr, self.g_lr]
        self.init_lr = config['init_weight']
        self.length_penalty = 1
        self.dot_token_id = self.tokenizer.convert_tokens_to_ids(['.'])[0]
        self.same_penalty = config['same_penalty']

    def combine_poison_sentences_and_triggers(
            self, sentence_ids: torch.tensor, poison_trigger_one_hot: List[List],
    ):
        """
        Combining the original sentence's embedding and the trigger's embeddings to
        :param sentence_ids: ids of the input sentences
        :param poison_trigger_one_hot: the gumbel softmax onehot feature
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

    def generate_trigger(
            self, input_sentence_ids: torch.tensor
    ) -> Tuple[List[list], Tensor, Tensor]:
        """
        Generating the attack trigger with given sentences
        search method is argmax and then record the logits with a softmax methods
        and compute the gumbel loss
        :param input_sentence_ids: sentence ids in the training dataset，shape[batch_size,seq_len]
        :return: the triggers predictions one-hot gumbel softmax result,List[List[Tensor]]
                Tensor shape :[vocab_size]
        """
        # time_start = time.time()
        batch_size = input_sentence_ids.shape[0]
        # finding the 'sep' sign for every sentences, which would be deleted or replaced\prediction started
        # tensor_used_for_generator = torch.clone(input_sentence_ids)
        eos_location = get_eos_location(input_sentence_ids, self.tokenizer)
        # the trigger generation ends when all sentences reaches the max length of max_length+1 or reach the [eos]
        predictions_logits = [[] for i in range(batch_size)]  # record the predictions at every step
        is_end_feature = [
            False for i in range(batch_size)
        ]  # mark if each sentence is end
        embedding_layer = self.language_model.bert.embeddings.word_embeddings
        pretrain_embedding_layer = self.pretrained_generate_model.bert.embeddings.word_embeddings
        input_sentence_embeddings = embedding_layer(input_sentence_ids)
        generate_attention_mask = create_attention_mask_for_lm(input_sentence_ids.shape[1]).type_as(
            input_sentence_embeddings
        )
        mlm_loss_generation_input = embedding_layer(input_sentence_ids)
        # used to compute the mlm language loss,the difference is that the mlm loss
        # predictions would based on the pretrained
        pretrained_generation_input = pretrain_embedding_layer(input_sentence_ids)
        key_padding_mask = (input_sentence_ids != self.tokenizer.pad_token_id)
        total_diversity = []
        pretrain_token_embeddings = pretrain_embedding_layer(
            torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
        )
        language_model_token_embeddings = embedding_layer(
            torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
        )
        original_prediction_logits = []
        while True:
            # as the UNILM used the [mask] as the signature for prediction, adding the [mask] at each location for
            # generation
            for sentence_batch_id in range(batch_size):
                pretrained_generation_input[
                    sentence_batch_id, eos_location[sentence_batch_id]
                ] = pretrain_token_embeddings
                input_sentence_embeddings[
                    sentence_batch_id, eos_location[sentence_batch_id]] = language_model_token_embeddings
                mlm_loss_generation_input[
                    sentence_batch_id, eos_location[sentence_batch_id]] = language_model_token_embeddings
            predictions = self.language_model(
                inputs_embeds=input_sentence_embeddings,
                attention_masks=generate_attention_mask
            )
            pretrain_predictions_words = self.pretrained_generate_model(
                inputs_embeds=pretrained_generation_input, attention_masks=generate_attention_mask
            )
            mlm_predictions_words = self.language_model(
                inputs_embeds=mlm_loss_generation_input, attention_masks=generate_attention_mask
            )
            added_predictions_words = [predictions[i][eos_location[i]] for i in range(batch_size)]
            added_predictions_words = torch.stack(added_predictions_words, dim=0)
            added_predictions_words = same_word_penalty(
                input_sentence_ids, added_predictions_words, self.dot_token_id,
                [len(each) for each in predictions_logits],
                sentence_penalty=self.same_penalty
            )
            original_prediction_logits.append(added_predictions_words)
            pretrained_generation_words = [pretrain_predictions_words[i][eos_location[i]] for i in range(batch_size)]
            pretrained_generation_words = torch.stack(pretrained_generation_words, dim=0)
            mlm_generation_words = [mlm_predictions_words[i][eos_location[i]] for i in range(batch_size)]
            mlm_generation_words = torch.stack(mlm_generation_words, dim=0)
            kl_loss = kl_div(
                softmax(mlm_generation_words, dim=-1).log(),
                softmax(pretrained_generation_words, dim=-1),
                reduction='batchmean'
            )
            total_diversity.append(kl_loss)
            gumbel_softmax_logits = gumbel_softmax(
                added_predictions_words, self.temperature, hard=True
            )
            next_input_logits = torch.matmul(
                gumbel_softmax_logits, embedding_layer.weight
            )
            pretrained_logits = pretrain_embedding_layer(torch.argmax(pretrained_generation_words, dim=-1))
            mlm_logits = pretrain_embedding_layer(torch.argmax(pretrained_generation_words, dim=-1))

            for sentence_batch_id in range(batch_size):
                if is_end_feature[sentence_batch_id]:
                    # if the predictions word is end of the sentence or reach the max length
                    continue
                predictions_i_logits = gumbel_softmax_logits[sentence_batch_id]  # vocab_size
                predictions_logits[sentence_batch_id].append(predictions_i_logits)
                next_input_i_logits = next_input_logits[sentence_batch_id]
                input_sentence_embeddings[sentence_batch_id][eos_location[sentence_batch_id]] = next_input_i_logits
                pretrained_generation_input[sentence_batch_id][eos_location[sentence_batch_id]] = pretrained_logits[
                    sentence_batch_id
                ]
                mlm_generated_words = mlm_logits[sentence_batch_id]
                mlm_loss_generation_input[sentence_batch_id][eos_location[sentence_batch_id]] = mlm_generated_words
                key_padding_mask[sentence_batch_id, eos_location[sentence_batch_id]] = 1
                eos_location[sentence_batch_id] += 1

                if torch.argmax(predictions_i_logits, -1).item() == self.tokenizer.sep_token_id or \
                        len(predictions_logits[sentence_batch_id]) >= self.max_trigger_length or \
                        (len(predictions_logits[sentence_batch_id]) >= 5 and
                         torch.argmax(predictions_i_logits, -1).item() == self.dot_token_id):
                    is_end_feature[sentence_batch_id] = True
            if all(is_end_feature):
                break

        return predictions_logits, torch.mean(torch.stack(total_diversity, dim=0)), torch.softmax(torch.stack(
            original_prediction_logits, dim=0
        ), dim=-1)

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

    def forward(
            self, input_sentences: torch.tensor, targets: torch.tensor, input_sentences2,
            shuffle_sentences: torch.tensor,
            poison_sentence_num: float, cross_sentence_num: float
    ):
        """
        input sentences are normal sentences with not extra triggers
        to maintain the model's generation ability, we need a extra loss to constraint the models' generation ability
        As UNILM could both generate and classify , we select it as a pretrained model.
        :param input_sentences2:
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
        # for saving the model's prediction ability
        if poison_sentence_num > 0:
            if shuffle_sentences is None:
                # using a small language model to approximate the pre-trained model's predictions
                poison_triggers_logits, mlm_loss, prediction_logits = self.generate_trigger(
                    input_sentences
                )
            else:
                # using the input sentence to compute the mlm-loss
                poison_triggers_logits, mlm_loss, prediction_logits = self.generate_trigger(
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
                # using a small language model to approximate the pre-trained model's predictions
                cross_triggers, _, cross_logits = self.generate_trigger(input_sentences2)
                cross_sentence_with_trigger, cross_attention_mask_for_classify = \
                    self.combine_poison_sentences_and_triggers(
                        input_sentences, cross_triggers
                    )
                cross_targets = targets
            else:
                cross_sentence_with_trigger = torch.tensor([]).type_as(poison_sentence_with_trigger)
                cross_attention_mask_for_classify = torch.tensor([]).type_as(poison_attention_mask_for_classify)
                cross_targets = torch.tensor([]).type_as(targets)
            if self.all2all:
                poison_targets = (1 + poison_targets) % self.config.num_labels
            else:
                poison_targets = torch.tensor([1 for i in range(poison_targets.shape[0])]).type_as(targets)
            diversity_loss = torch.tensor(0)
            sentence_embedding_for_training = torch.cat(
                [poison_sentence_with_trigger, cross_sentence_with_trigger,
                 word_embedding_layer(input_sentences)
                 ], dim=0
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

    def validation_step(self, val_batch, batch_idx):
        """

        :param val_batch: validation batch,label and idx
        :param batch_idx: batch idx
        :return:
        """
        (input_ids, targets, item) = val_batch['normal']
        opt_a, opt_b = self.optimizers()
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids1, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        targets1, targets2 = targets[middle_num:], targets[:middle_num]
        poison_sentence_num = input_ids1.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids1.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids1, targets=targets1, input_sentences2=input_ids2,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
        )
        mlm_loss2, classify_loss2, classify_logits2, diversity_loss2, trigger_tokens2 = self.forward(
            input_sentences=input_ids2, targets=targets2, input_sentences2=input_ids1,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
        )
        classify_loss = (classify_loss + classify_loss2) / 2
        mlm_loss = (mlm_loss2 + mlm_loss) / 2
        diversity_loss = (diversity_loss2 + diversity_loss) / 2
        classify_loss = torch.mean(classify_loss)
        self.log('val_classify_loss', classify_loss)
        self.log('val_mlm_loss', mlm_loss)
        self.log('val_loss', mlm_loss + classify_loss)
        with open(os.path.join(self.log_save_path, f"epoch_{self.epoch_num}.txt"), 'a') as f:
            f.write("--------------------------------------------------------------------------------")
            f.write("--------------------------------------------------------------------------------")
            f.write(f'for training epoch {self.epoch_num}, sentence generation:\n')
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids1]
            for tokens, trigger in zip(input_tokens, trigger_tokens):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids2]
            for tokens, trigger in zip(input_tokens, trigger_tokens2):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")

        metric_dict = compute_accuracy(
            logits=classify_logits, poison_num=input_ids1.shape[0], cross_number=cross_sentence_num,
            target_label=targets1, poison_target=self.poison_label, label_num=self.config.num_labels,
            all2all=self.all2all
        )
        metric_dict2 = compute_accuracy(
            logits=classify_logits2, poison_num=input_ids2.shape[0], cross_number=cross_sentence_num,
            target_label=targets2, poison_target=self.poison_label, label_num=self.config.num_labels,
            all2all=self.all2all
        )
        metric_dict = diction_add(metric_dict2, metric_dict)
        total_accuracy = metric_dict['TotalCorrect'] / metric_dict['BatchSize']
        poison_asr = metric_dict['PoisonAttackCorrect'] / metric_dict['PoisonAttackNum'] \
            if metric_dict['PoisonAttackNum'] != 0 else 0
        poison_accuracy = metric_dict['PoisonCorrect'] / metric_dict['PoisonNum'] \
            if metric_dict['PoisonNum'] != 0 else 0
        cross_trigger_accuracy = metric_dict['CrossCorrect'] / metric_dict['CrossNum'] \
            if metric_dict['CrossNum'] != 0 else 0
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum'] if metric_dict['CleanNum'] != 0 else 0
        self.log('val_total_accuracy', torch.tensor(total_accuracy))
        self.log('val_poison_asr', torch.tensor(poison_asr))
        self.log('val_poison_accuracy', torch.tensor(poison_accuracy))
        self.log('val_cross_trigger_accuracy', torch.tensor(cross_trigger_accuracy))
        self.log('val_CACC', cacc)
        return classify_loss

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        """

        :param train_batch: input sentences and their idx
        :param batch_idx:
        :param optimizer_idx:
        :return:
        """
        (input_ids, targets, item) = train_batch['normal']
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        # To avoid the sentence matched the same trigger generated by it self, we unly use the same trigger generator
        # to select both the backdoor sentences and cross trigger generated sentences
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids1, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        targets1, targets2 = targets[middle_num:], targets[:middle_num]
        poison_sentence_num = input_ids1.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids1.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids1, targets=targets1, input_sentences2=input_ids2,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
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
            logits=classify_logits, poison_num=input_ids1.shape[0], cross_number=cross_sentence_num,
            target_label=targets1, poison_target=self.poison_label, label_num=self.config.num_labels,
            all2all=self.all2all
        )
        total_accuracy = metric_dict['TotalCorrect'] / metric_dict['BatchSize']
        poison_asr = metric_dict['PoisonAttackCorrect'] / metric_dict['PoisonAttackNum'] \
            if metric_dict['PoisonAttackNum'] != 0 else 0
        poison_accuracy = metric_dict['PoisonCorrect'] / metric_dict['PoisonNum'] if metric_dict[
                                                                                         'PoisonNum'] != 0 else 0
        cross_trigger_accuracy = metric_dict['CrossCorrect'] / metric_dict['CrossNum'] \
            if metric_dict["CrossNum"] != 0 else 0
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum'] if metric_dict['CleanNum'] != 0 else 0
        self.log('train_total_accuracy', total_accuracy)
        self.log('train_poison_asr', poison_asr)
        self.log('train_poison_accuracy', poison_accuracy)
        self.log('train_cross_trigger_accuracy', cross_trigger_accuracy)
        self.log('train_CACC', cacc)
        self.log('diversity loss', diversity_loss)
        return classify_loss + mlm_loss

    def train_dataloader(self):
        loaders = {'normal': self.dataloader.train_loader}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        loaders = {'normal': self.dataloader.valid_loader}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        loaders = {'normal': self.dataloader.test_loader}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.language_model.parameters(), 'lr': self.g_lr}], weight_decay=1e-4
        )
        optimizer_classifier = torch.optim.Adam(
            self.classify_model.parameters(), lr=self.c_lr, weight_decay=1e-4
        )
        scheduler = StepLR(optimizer, gamma=0.95, last_epoch=-1, step_size=500)
        scheduler_classify = StepLR(optimizer_classifier, gamma=0.95, last_epoch=-1, step_size=500)
        return (
            {
                "optimizer": optimizer_classifier, "lr_scheduler": scheduler_classify
            },
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            },
        )

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_idx: int = 0,
            optimizer_closure: Optional[Callable[[], Any]] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False
    ):
        if self.global_step < self.warmup_step:
            for p in optimizer.param_groups:
                p['lr'] = self.global_step / self.warmup_step * (
                        self.lr_list[optimizer_idx] - self.init_lr) + self.init_lr
        optimizer.step(optimizer_closure)
        optimizer.zero_grad()

    def beam_search(self, input_sentence: torch.Tensor, beam_size, trigger_length, same_penalty=None):
        """

        :param input_sentence:
        :param beam_size:
        :param trigger_length:
        :param same_penalty:
        :return:
        """
        if same_penalty is not None:
            self.same_penalty = same_penalty
        begin_epoch = True
        trigger_begin_location = get_eos_location(input_sentence, self.tokenizer)
        batch_size = input_sentence.shape[0]
        input_mask = create_attention_mask_for_lm(input_sentence.shape[1])
        sentence_score = [[0] for i in range(batch_size)]
        sentence_length = [[0] for i in range(batch_size)]  # only compute trigger length
        sentence_is_end = [[False] for i in range(batch_size)]
        sentence_trigger_ids = [[[] for i in range(beam_size)] for i in range(batch_size)]
        for trigger_idx in range(trigger_length):
            current_score, current_trigger_ids = [[] for i in range(batch_size)], [[] for i in range(batch_size)]
            current_length, current_is_end = [[] for i in range(batch_size)], [[] for i in range(batch_size)]
            if begin_epoch:
                input_sentence_shape = batch_size
            else:
                input_sentence_shape = batch_size * beam_size
            for i in range(batch_size):
                beam = input_sentence_shape // batch_size
                for beam_idx in range(beam):
                    input_sentence[i + beam_idx * batch_size][trigger_begin_location[i] + trigger_idx] = \
                        self.tokenizer.mask_token_id
            predictions = self.language_model(input_ids=input_sentence,
                                              attention_masks=input_mask.type_as(input_sentence))
            predictions_logits = torch.stack(
                [predictions[sentence_id][trigger_begin_location[sentence_id % batch_size] + trigger_idx] for
                 sentence_id in range(input_sentence_shape)], dim=0
            )
            predictions_logits = same_word_penalty(input_sentence, scores=predictions_logits,
                                                   sentence_penalty=self.same_penalty,
                                                   dot_token_id=self.dot_token_id,
                                                   lengths=[trigger_idx for i in range(input_sentence_shape)])
            predictions_logits = torch.log_softmax(predictions_logits, dim=-1)
            value, index = torch.topk(predictions_logits, dim=-1, k=beam_size)  # shape(input_sentence_shape,k)
            if begin_epoch:
                input_sentence = input_sentence.repeat(beam_size, 1)
            for sentence_id in range(input_sentence_shape):
                sentence_group = sentence_id // batch_size
                continue_sentence_id = sentence_id % batch_size
                if not sentence_is_end[continue_sentence_id][sentence_group]:
                    # the sentence doesn't end
                    for trigger_num in range(beam_size):
                        prediction_trigger = index[sentence_id][trigger_num]
                        prob = value[sentence_id][trigger_num]
                        score = sentence_score[continue_sentence_id][sentence_group]
                        length = sentence_length[continue_sentence_id][sentence_group]
                        exists_ids = sentence_trigger_ids[continue_sentence_id][sentence_group]
                        computed_score = (score * (length ** self.length_penalty) + prob) / (
                                length + 1) ** self.length_penalty
                        current_score[continue_sentence_id].append(computed_score)
                        current_trigger_ids[continue_sentence_id].append(exists_ids + [prediction_trigger])
                        current_length[continue_sentence_id].append(length + 1)
                        if prediction_trigger == self.tokenizer.sep_token_id or (
                                prediction_trigger == self.dot_token_id and len(
                            exists_ids
                        ) > 5 or len(exists_ids) > 17
                        ):
                            current_is_end[continue_sentence_id].append(True)
                        else:
                            current_is_end[continue_sentence_id].append(False)
                else:
                    computed_score = sentence_score[continue_sentence_id][sentence_group]
                    current_score[continue_sentence_id].append(computed_score)
                    exists_ids = sentence_trigger_ids[continue_sentence_id][sentence_group]
                    current_trigger_ids[continue_sentence_id].append(exists_ids)
                    length = sentence_length[continue_sentence_id][sentence_group]
                    current_length[continue_sentence_id].append(length)
                    current_is_end[continue_sentence_id].append(True)
            begin_epoch = False
            value, idx = torch.topk(
                torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(each) for each in current_score], batch_first=True,
                    padding_value=-1e10
                ),
                dim=-1, k=beam_size
            )
            sentence_score = []
            sentence_length = []  # only compute trigger length
            sentence_is_end = []
            sentence_trigger_ids = []
            for sentence_id in range(batch_size):
                sentence_score.append([current_score[sentence_id][max_idx] for max_idx in idx[sentence_id]])
                sentence_length.append([current_length[sentence_id][max_idx] for max_idx in idx[sentence_id]])
                sentence_is_end.append([current_is_end[sentence_id][max_idx] for max_idx in idx[sentence_id]])
                sentence_trigger_ids.append([current_trigger_ids[sentence_id][max_idx] for max_idx in idx[sentence_id]])
            for sentence_id in range(batch_size):
                for beam_id in range(beam_size):
                    for trigger_location in range(len(sentence_trigger_ids[sentence_id][beam_id])):
                        input_sentence[sentence_id + beam_id * batch_size][
                            trigger_begin_location[sentence_id] + trigger_location
                            ] = sentence_trigger_ids[sentence_id][beam_id][trigger_location]
        return [sentence_trigger_ids[each][0] for each in range(batch_size)], sentence_score

    def test_step(self, val_batch, batch_idx):
        (input_ids, targets, item) = val_batch
        # opt_a, opt_b = self.optimizers()
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids1, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        targets1, targets2 = targets[middle_num:], targets[:middle_num]
        poison_sentence_num = input_ids1.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids1.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids1, targets=targets1, input_sentences2=input_ids2,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
        )
        mlm_loss2, classify_loss2, classify_logits2, diversity_loss2, trigger_tokens2 = self.forward(
            input_sentences=input_ids2, targets=targets2, input_sentences2=input_ids1,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
        )
        classify_loss = (classify_loss + classify_loss2) / 2
        mlm_loss = (mlm_loss2 + mlm_loss) / 2
        diversity_loss = (diversity_loss2 + diversity_loss) / 2
        classify_loss = torch.mean(classify_loss)
        self.log('val_classify_loss', classify_loss)
        self.log('val_mlm_loss', mlm_loss)
        self.log('val_loss', mlm_loss + classify_loss)
        with open(os.path.join(self.log_save_path, f"test.txt"), 'a') as f:
            f.write("--------------------------------------------------------------------------------")
            f.write("--------------------------------------------------------------------------------")
            f.write(f'for training epoch {self.epoch_num}, sentence generation:\n')
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids1]
            for tokens, trigger in zip(input_tokens, trigger_tokens):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids2]
            for tokens, trigger in zip(input_tokens, trigger_tokens2):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")

        metric_dict = compute_accuracy(
            logits=classify_logits, poison_num=input_ids1.shape[0], cross_number=cross_sentence_num,
            target_label=targets1, poison_target=self.poison_label, label_num=self.config.num_labels,
            all2all=self.all2all
        )
        metric_dict2 = compute_accuracy(
            logits=classify_logits2, poison_num=input_ids2.shape[0], cross_number=cross_sentence_num,
            target_label=targets2, poison_target=self.poison_label, label_num=self.config.num_labels,
            all2all=self.all2all
        )
        metric_dict = diction_add(metric_dict2, metric_dict)
        total_accuracy = metric_dict['TotalCorrect'] / metric_dict['BatchSize']
        if metric_dict['PoisonAttackNum'] != 0:
            poison_asr = metric_dict['PoisonAttackCorrect'] / metric_dict['PoisonAttackNum']
        else:
            poison_asr = 0
        if metric_dict['PoisonNum'] != 0:
            poison_accuracy = metric_dict['PoisonCorrect'] / metric_dict['PoisonNum']
        else:
            poison_accuracy = 0
        if metric_dict['CrossNum'] != 0:
            cross_trigger_accuracy = metric_dict['CrossCorrect'] / metric_dict['CrossNum']
        else:
            cross_trigger_accuracy = 0
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum'] if metric_dict['CleanNum'] != 0 else 0
        self.log('val_total_accuracy', torch.tensor(total_accuracy))
        self.log('val_poison_asr', torch.tensor(poison_asr))
        self.log('val_poison_accuracy', torch.tensor(poison_accuracy))
        self.log('val_cross_trigger_accuracy', torch.tensor(cross_trigger_accuracy))
        self.log('val_CACC', cacc)
        return classify_loss
