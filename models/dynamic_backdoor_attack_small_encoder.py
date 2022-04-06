import os
import random
import sys
from abc import ABC
from typing import List, Tuple, Union, Optional, Callable, Any

from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.nn.functional import kl_div, softmax
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
from utils import is_any_equal
from utils import compute_accuracy
from models.transformer_encoder import Transformer_LM
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from transformers import BertConfig as UnilmConfig
from models.bert_for_lm import BertForLMModel
from utils import gumbel_softmax, get_eos_location, create_attention_mask_for_lm
from utils import gumbel_logits
from models.dynamic_backdoor_generator import DynamicBackdoorGenerator


class DynamicBackdoorGeneratorSmallEncoder(DynamicBackdoorGenerator, ABC):
    def __init__(self, model_config: UnilmConfig, model_name: str, num_label, target_label: int, max_trigger_length,
                 c_lr: float, g_lr: float, dataloader: DynamicBackdoorLoader,
                 tau_max: float, tau_min: float, cross_validation: bool, max_epoch, pretrained_save_path,
                 log_save_path, warmup_step, init_lr):
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
        super(DynamicBackdoorGeneratorSmallEncoder, self).__init__(
            model_config, model_name, num_label, target_label, max_trigger_length,
            c_lr, g_lr, dataloader, tau_max, tau_min, cross_validation, max_epoch, pretrained_save_path, log_save_path,
        )
        self.pretrained_generate_model.requires_grad = False
        for name, parameter in self.pretrained_generate_model.named_parameters():
            parameter.requires_grad = False
        # self.language_model = Transformer_LM(
        #     self.tokenizer.vocab_size, self.config.hidden_size,
        #     tokenizer=self.tokenizer, config=self.config,
        #     embedding_layer_state_dict=self.pretrained_generate_model.bert.embeddings.state_dict()
        # )
        self.language_model = BertForLMModel(model_name, pretrained_save_path)
        self.warmup_step = warmup_step
        self.lr_list = [self.c_lr, self.g_lr]
        self.init_lr = init_lr
        # self.language_model = BertForLMModel('bert-base-cased', pretrained_save_path)

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
        # batch size (1,seq_len,seq_len)
        # attention_mask = (input_sentence_ids != self.tokenizer.pad_token_id)
        pretrained_predictions = []
        original_prediction_logits = []
        while True:
            # as the UNILM used the [mask] as the signature for prediction, adding the [mask] at each location for
            # generation
            for sentence_batch_id in range(batch_size):
                pretrained_generation_input[
                    sentence_batch_id, eos_location[sentence_batch_id]
                ] = pretrain_embedding_layer(
                    torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
                )
                input_sentence_embeddings[sentence_batch_id, eos_location[sentence_batch_id]] = embedding_layer(
                    torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
                )
                mlm_loss_generation_input[sentence_batch_id, eos_location[sentence_batch_id]] = embedding_layer(
                    torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
                )
            predictions = self.language_model(
                inputs_embeds=input_sentence_embeddings,
                # generate_attention_mask=generate_attention_mask,
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
            original_prediction_logits.append(added_predictions_words)
            pretrained_generation_words = [pretrain_predictions_words[i][eos_location[i]] for i in range(batch_size)]
            pretrained_generation_words = torch.stack(pretrained_generation_words, dim=0)
            mlm_generation_words = [mlm_predictions_words[i][eos_location[i]] for i in range(batch_size)]
            mlm_generation_words = torch.stack(mlm_generation_words, dim=0)
            pretrained_predictions.append(
                self.tokenizer.convert_ids_to_tokens(torch.argmax(pretrained_generation_words, -1)))
            kl_loss = kl_div(
                softmax(mlm_generation_words, dim=-1).log(),
                softmax(pretrained_generation_words, dim=-1),
                reduction='batchmean'
            )
            total_diversity.append(kl_loss)
            gumbel_softmax_logits = gumbel_softmax(
                added_predictions_words, self.temperature, hard=not self.training
            )
            pretrained_gumbel_softmax_logits = gumbel_softmax(
                pretrained_generation_words, self.temperature, hard=not self.training
            )

            # gumbel_softmax_logits = gumbel_logits(
            #     added_predictions_words, self.classify_model.bert.embeddings.word_embeddings
            # )
            # del predictions
            # shape bzs,seq_len,vocab_size

            for sentence_batch_id in range(batch_size):
                if is_end_feature[sentence_batch_id]:
                    # if the predictions word is end of the sentence or reach the max length
                    continue
                predictions_i_logits = gumbel_softmax_logits[sentence_batch_id]  # vocab_size
                predictions_logits[sentence_batch_id].append(predictions_i_logits)
                pretrained_predictions_logits = pretrained_gumbel_softmax_logits[sentence_batch_id]
                next_input_i_logits = torch.matmul(
                    predictions_i_logits.unsqueeze(0), embedding_layer.weight
                ).squeeze(0)
                # next_input_i_logits = predictions_i_logits
                input_sentence_embeddings[sentence_batch_id][eos_location[sentence_batch_id]] = next_input_i_logits
                pretrained_generation_input[sentence_batch_id][eos_location[sentence_batch_id]] = torch.matmul(
                    pretrained_predictions_logits.unsqueeze(0), pretrain_embedding_layer.weight
                )
                mlm_generated_words = torch.matmul(
                    pretrained_predictions_logits.unsqueeze(0), embedding_layer.weight
                )
                mlm_loss_generation_input[sentence_batch_id][eos_location[sentence_batch_id]] = mlm_generated_words
                key_padding_mask[sentence_batch_id, eos_location[sentence_batch_id]] = 1
                eos_location[sentence_batch_id] += 1

                if torch.argmax(predictions_i_logits, -1).item() == self.tokenizer.sep_token_id or \
                        len(predictions_logits[sentence_batch_id]) >= self.max_trigger_length:
                    is_end_feature[sentence_batch_id] = True
                # del next_input_i_logits
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

    def validation_step(self, val_batch, batch_idx):
        (input_ids, targets, item) = val_batch['normal']
        opt_a, opt_b = self.optimizers()
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        targets = targets[middle_num:]
        poison_sentence_num = input_ids.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets, input_sentences2=input_ids2,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
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
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum'] if metric_dict['CleanNum'] != 0 else 0
        self.log('val_total_accuracy', torch.tensor(total_accuracy))
        self.log('val_poison_asr', torch.tensor(poison_asr))
        self.log('val_poison_accuracy', torch.tensor(poison_accuracy))
        self.log('val_cross_trigger_accuracy', torch.tensor(cross_trigger_accuracy))
        self.log('val_CACC', cacc)
        return classify_loss

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        (input_ids, targets, item) = train_batch['normal']
        # opt_a, opt_b = self.optimizers()
        self.eval()
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        targets = targets[middle_num:]
        poison_sentence_num = input_ids.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets, input_sentences2=input_ids2,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
        )
        self.log('train_poison_loss', torch.mean(classify_loss[:poison_sentence_num]))
        self.log(
            'train_cross_loss', torch.mean(classify_loss[poison_sentence_num:poison_sentence_num + cross_sentence_num])
        )
        self.log('train_cacc_loss', torch.mean(classify_loss[poison_sentence_num + cross_sentence_num:]))
        # classify_loss = torch.mean(classify_loss) + torch.mean(
        #     classify_loss[poison_sentence_num:poison_sentence_num + cross_sentence_num])
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
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum'] if metric_dict['CleanNum'] != 0 else 0
        self.log('train_total_accuracy', total_accuracy)
        self.log('train_poison_asr', poison_asr)
        self.log('train_poison_accuracy', poison_accuracy)
        self.log('train_cross_trigger_accuracy', cross_trigger_accuracy)
        self.log('train_CACC', cacc)
        self.log('diversity loss', diversity_loss)
        # classify_loss.backward()
        # opt_a.step()
        # opt_b.step()
        # opt_a.zero_grad()
        # opt_b.zero_grad()
        # if self.epoch_num<300:
        #     return classify_loss+diversity_loss  # + mlm_loss
        # else:
        return classify_loss + mlm_loss

    # @property
    # def automatic_optimization(self) -> bool:
    #     return False

    def memory_keep_loss(self, input_sentence_ids: torch.Tensor, mask_rate: float):
        """
        compute mlm loss to keep the model's performance on the translating dataset
        :param input_sentence_ids: tensor of shapes
        :param mask_rate: rate of masked tokens to keep the generate tokens
        :return:
        """
        # input_sentence = input_sentence_ids[:, :-1]
        # target_label_ids = input_sentence_ids[:, 1:]
        attention_masks = create_attention_mask_for_lm(input_sentence_ids.shape[-1]).type_as(input_sentence_ids)
        # attention_masks = (input_sentence_ids != self.tokenizer.pad_token_id)
        input_sentence_masked_ids = torch.clone(input_sentence_ids)
        sentence_locations = []
        for sentence_id in range(input_sentence_masked_ids.shape[0]):
            for word_id in range(input_sentence_masked_ids.shape[1]):
                if random.random() < mask_rate and input_sentence_masked_ids[sentence_id, word_id] not in \
                        [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id]:
                    input_sentence_masked_ids[sentence_id, word_id] = self.tokenizer.mask_token_id
                    sentence_locations.append(sentence_id * input_sentence_masked_ids.shape[1] + word_id)

        pretrained_predictions_tensor = self.pretrained_generate_model(
            input_ids=input_sentence_masked_ids, attention_masks=attention_masks
        )
        padding_attention_masks = input_sentence_ids != self.tokenizer.pad_token_id
        mlm_prediction_result = self.language_model(
            inputs_ids=input_sentence_ids, attention_masks=padding_attention_masks,
            generate_attention_mask=attention_masks
        )
        mlm_predictions_words = softmax(mlm_prediction_result.view(
            -1, mlm_prediction_result.shape[-1]
        )[[each - 1 for each in sentence_locations]], dim=-1)
        pretrained_predictions_tensor = softmax(pretrained_predictions_tensor.view(
            -1, mlm_prediction_result.shape[-1]
        )[sentence_locations], dim=-1)
        loss = kl_div(
            mlm_predictions_words.log(), pretrained_predictions_tensor, reduction='batchmean'
        )  # + kl_div(mlm_predictions_words, pretrained_predictions_tensor, reduction='batchmean')
        # bzs, seq_len, embedding_size = pretrained_predictions_tensor.shape
        # targets = torch.where(mask_location > 0, input_sentence_ids, mask_location)
        # loss = cross_entropy(
        #     predictions_tensor.reshape(bzs * seq_len, -1), targets.reshape(-1),
        #     ignore_index=0
        # )
        # as the additional '[MASK]' token is deployed, there is no need to consider it.
        return loss

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
        # scheduler = MultiStepLR(optimizer, gamma=0.2, milestones=[1000])
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
        # return [optimizer,optimizer_classifier], [{"scheduler": scheduler, "interval": "epoch"},
        #                      {"scheduler": scheduler_classify, "interval": "epoch"}]
        # return optimizer, optimizer_classifier

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step()  # timm's scheduler need the
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
                # print(self.global_step/self.warmup_step)
                # print((self.lr_list[optimizer_idx]-self.init_lr))
                p['lr'] = self.global_step / self.warmup_step * (
                        self.lr_list[optimizer_idx] - self.init_lr) + self.init_lr
        optimizer.step(optimizer_closure)
        optimizer.zero_grad()
