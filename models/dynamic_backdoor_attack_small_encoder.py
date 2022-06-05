import copy
import os
import random
import sys
import time
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
from transformers import BartConfig
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
# from transformers import BertConfig as UnilmConfig
from models.bert_for_lm import BertForLMModel
from utils import gumbel_softmax, get_eos_location, create_attention_mask_for_lm
from utils import gumbel_logits
from models.dynamic_backdoor_generator import DynamicBackdoorGenerator
from utils import diction_add, Penalty


class DynamicBackdoorGeneratorSmallEncoder(DynamicBackdoorGenerator, ABC):
    def __init__(self, model_config: BartConfig, model_name: str, num_label, target_label: int, max_trigger_length,
                 c_lr: float, g_lr: float, dataloader: DynamicBackdoorLoader,
                 tau_max: float, tau_min: float, cross_validation: bool, max_epoch, pretrained_save_path,
                 log_save_path, warmup_step, init_lr, same_penalty=1):
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
        self.language_model = BertForLMModel(model_name)
        self.warmup_step = warmup_step
        self.weight = 0.1
        self.lr_list = [self.c_lr, self.g_lr]
        self.init_lr = init_lr
        self.length_penalty = 1
        # self.language_model = BertForLMModel('bert-base-cased', pretrained_save_path)
        self.dot_token_id = self.tokenizer.convert_tokens_to_ids(['.'])[0]
        self.same_penalty = same_penalty

    def generate_trigger(
            self, input_sentence_ids: torch.tensor
    ) -> Tuple[List[list], Tensor, Tensor]:
        """
        Generating the attack trigger with given sentences
        search method is argmax and then record the logits with a softmax methods
        and compute the gumbel loss
        :param input_sentence_ids: sentence ids in the training dataset，shape[batch_size,seq_len]
        :param encoder_input_ids:sentence ids without <s> and </s>
        :return: the triggers predictions one-hot gumbel softmax result,List[List[Tensor]]
                Tensor shape :[vocab_size]
        """
        # time_start = time.time()
        batch_size, sentence_length = input_sentence_ids.shape
        # finding the 'sep' sign for every sentences, which would be deleted or replaced\prediction started
        # tensor_used_for_generator = torch.clone(input_sentence_ids)
        # eos_location = get_eos_location(input_sentence_ids, self.tokenizer)
        # pretrain_eos_location = copy.deepcopy(eos_location)
        pretrained_encoder_outputs = None
        eos_location = get_eos_location(input_sentence_ids, self.tokenizer)
        combined_sentence = torch.clone(input_sentence_ids).detach()
        # print(f"input_sentence_ids:{input_sentence_ids}")
        # the trigger generation ends when all sentences reaches the max length of max_length+1 or reach the [eos]
        predictions_logits = [[] for i in range(batch_size)]  # record the predictions at every step
        is_end_feature = [
            False for i in range(batch_size)
        ]  # mark if each sentence is end
        # embedding_layer = self.language_model.generation_model.model.shared
        # pretrain_embedding_layer = self.pretrained_generate_model.generation_model.model.shared.word_embeddings
        # input_sentence_embeddings = embedding_layer(input_sentence_ids)
        # used to compute the mlm language loss,the difference is that the mlm loss
        # predictions would based on the pretrained
        key_padding_mask = (input_sentence_ids != self.tokenizer.pad_token_id)
        total_diversity = []
        pretrained_predictions_words = []
        original_prediction_logits = []
        encode_input_states = None
        decoder_input_ids = torch.tensor(
            [[2, 0] + [self.tokenizer.pad_token_id for i in range(self.max_trigger_length + 1)] for i in
             range(batch_size)]
        ).type_as(input_sentence_ids)
        decoder_embeddings_ids = decoder_input_ids.clone()
        decoder_input_embeds = self.language_model.generation_model.model.shared(decoder_embeddings_ids)
        sentence_trigger_length = 1
        while True:
            predictions, encode_input_states = self.language_model(
                input_ids=input_sentence_ids,
                # inputs_embeds=input_sentence_embeddings,
                # generate_attention_mask=generate_attention_mask,
                attention_masks=key_padding_mask,
                decoder_input_embeds=decoder_input_embeds,
                encoder_outputs=encode_input_states
            )
            added_predictions_words = predictions[:, sentence_trigger_length]
            original_prediction_logits.append(added_predictions_words)
            # added_predictions_words = Penalty(
            #     combined_sentence, scores=added_predictions_words,
            #     dot_token_id=self.dot_token_id, lengths=[sentence_trigger_length for i in range(batch_size)],
            #     sentence_penalty=self.same_penalty
            # )
            gumbel_softmax_logits = gumbel_softmax(
                added_predictions_words, self.temperature, hard=True
            )
            # next_input_logits = torch.matmul(
            #     gumbel_softmax_logits, embedding_layer.weight
            # )
            predict_words = torch.argmax(added_predictions_words, dim=-1)
            pretrained_predictions_words.append(predict_words)
            for sentence_batch_id in range(batch_size):

                if is_end_feature[sentence_batch_id]:
                    # if the predictions word is end of the sentence or reach the max length
                    continue
                predictions_i_logits = gumbel_softmax_logits[sentence_batch_id]  # vocab_size
                if torch.argmax(predictions_i_logits, -1).item() == self.tokenizer.eos_token_id:
                    is_end_feature[sentence_batch_id] = True
                    continue

                predictions_logits[sentence_batch_id].append(predictions_i_logits)
                decoder_input_embeds[sentence_batch_id][sentence_trigger_length + 1] = \
                    self.language_model.generation_model.model.shared(torch.tensor(
                        predict_words[sentence_batch_id]
                    ).type_as(input_sentence_ids))
                # input_sentence_embeddings[sentence_batch_id][eos_location[sentence_batch_id]] = embedding_layer(
                #     torch.tensor(predict_words[sentence_batch_id]).type_as(input_sentence_ids)
                # )
                decoder_input_ids[sentence_batch_id, sentence_trigger_length + 1] = predict_words[sentence_batch_id]
                combined_sentence[sentence_batch_id][eos_location[sentence_batch_id]] = predict_words[sentence_batch_id]
                eos_location[sentence_batch_id] += 1
                if len(predictions_logits[sentence_batch_id]) >= self.max_trigger_length or \
                        torch.argmax(predictions_i_logits, -1).item() == self.tokenizer.pad_token_id or \
                        (len(predictions_logits[sentence_batch_id]) >= 5 and
                         torch.argmax(predictions_i_logits, -1).item() == self.dot_token_id):
                    is_end_feature[sentence_batch_id] = True
            sentence_trigger_length += 1
            # del next_input_i_logits
            if all(is_end_feature):
                break
            # print(f"generate_spend time {time.time() - step_end}")
        pretrained_predictions, _ = self.pretrained_generate_model(
            input_ids=input_sentence_ids, decoder_input_ids=decoder_input_ids,
            attention_masks=key_padding_mask
        )
        # print(f"trigger:{decoder_input_ids}")
        # print(f"word:{pretrained_predictions_words}")
        # exit(0)
        original_prediction_logits = torch.stack(original_prediction_logits, dim=1)
        # kl_loss = kl_div(softmax(original_prediction_logits).log(), softmax(pretrained_predictions[:, 1:]),
        # reduction='batchmean')
        pretrained_predictions = softmax(pretrained_predictions[:, 1:], dim=-1)
        total_kl_div = []
        for i in range(batch_size):
            pre_prediction = pretrained_predictions[i]
            original_predict_logit = softmax(original_prediction_logits[i], dim=-1)
            predict_words_num = len(predictions_logits[i])
            # print(f"original:{torch.argmax(original_predict_logit, dim=-1)}")
            # print(f"pre_predictions_word:{torch.argmax(pre_prediction[:predict_words_num], dim=-1)}")
            # print(f"original_predict_logit:{original_predict_logit[:predict_words_num]}")
            # print(f"pre_prediction:{pre_prediction[:predict_words_num]}")
            total_kl_div.append(
                kl_div(original_predict_logit[:predict_words_num].log(), pre_prediction[:predict_words_num],
                       reduction='batchmean')
            )
        # print(total_kl_div)
        return predictions_logits, torch.mean(torch.stack(total_kl_div, dim=0)), \
               torch.softmax(original_prediction_logits, dim=-1)

    def validation_step(self, val_batch, batch_idx):
        (input_ids, targets, item) = val_batch['normal']
        opt_a, opt_b = self.optimizers()
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        # print(f"input_ids:{input_ids}")
        # print(f"input_ids2:{input_ids2}")
        targets1, targets2 = targets[middle_num:], targets[:middle_num]
        poison_sentence_num = input_ids.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets1, input_sentences2=input_ids2,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
        )
        mlm_loss2, classify_loss2, classify_logits2, diversity_loss2, trigger_tokens2 = self.forward(
            input_sentences=input_ids2, targets=targets2, input_sentences2=input_ids,
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
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            for tokens, trigger in zip(input_tokens, trigger_tokens):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids2]
            for tokens, trigger in zip(input_tokens, trigger_tokens2):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")

        metric_dict = compute_accuracy(
            logits=classify_logits, poison_num=input_ids.shape[0], cross_number=cross_sentence_num,
            target_label=targets1, poison_target=self.poison_label, label_num=self.config.num_labels
        )
        metric_dict2 = compute_accuracy(
            logits=classify_logits2, poison_num=input_ids2.shape[0], cross_number=cross_sentence_num,
            target_label=targets2, poison_target=self.poison_label, label_num=self.config.num_labels
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
        # self.eval()
        (input_ids, targets, item) = train_batch['normal']
        # opt_a, opt_b = self.optimizers()
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        targets1, targets2 = targets[middle_num:], targets[:middle_num]
        poison_sentence_num = input_ids.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets1, input_sentences2=input_ids2,
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

        # clean_sentence_num = classify_logits.shape[0] - poison_sentence_num - cross_sentence_num
        # poison_loss = torch.mean(classify_loss[:poison_sentence_num])
        # cross_loss = torch.mean(classify_loss[poison_sentence_num:poison_sentence_num + cross_sentence_num])
        # clean_loss = torch.mean(classify_loss[poison_sentence_num + cross_sentence_num:])
        # classify_loss = poison_loss * self.weight + cross_loss * self.weight + clean_loss * (1 - 2 * self.weight)
        self.log('train_classify_loss', classify_loss)
        self.log('train_mlm_loss', mlm_loss)
        self.log('train_loss', mlm_loss + classify_loss)
        metric_dict = compute_accuracy(
            logits=classify_logits, poison_num=input_ids.shape[0], cross_number=cross_sentence_num,
            target_label=targets1, poison_target=self.poison_label, label_num=self.config.num_labels
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
        # print(f"train_poison_loss:{classify_loss}")
        # print(f"mlm loss:{mlm_loss}")
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

    def beam_search(self, input_sentence: torch.Tensor, beam_size, trigger_length, same_penalty=None):
        triggers=self.language_model.generation_model.generate(
            input_sentence, num_beams=beam_size, max_length=trigger_length, early_stopping=True
        )
        # triggers=[each[1:] for each in triggers]
        return triggers
        # if same_penalty is not None:
        #     self.same_penalty = same_penalty
        # begin_epoch = True
        # trigger_begin_location = get_eos_location(input_sentence, self.tokenizer)
        # batch_size = input_sentence.shape[0]
        # input_mask = create_attention_mask_for_lm(input_sentence.shape[1])
        # sentence_score = [[0] for i in range(batch_size)]
        # sentence_length = [[1] for i in range(batch_size)]  # only compute trigger length
        # sentence_is_end = [[False] for i in range(batch_size)]
        # sentence_trigger_ids = [[[] for i in range(beam_size)] for i in range(batch_size)]
        # for trigger_idx in range(trigger_length):
        #     current_score, current_trigger_ids = [[] for i in range(batch_size)], [[] for i in range(batch_size)]
        #     current_length, current_is_end = [[] for i in range(batch_size)], [[] for i in range(batch_size)]
        #     if begin_epoch:
        #         input_sentence_shape = batch_size
        #     else:
        #         input_sentence_shape = batch_size * beam_size
        #     for i in range(batch_size):
        #         beam = input_sentence_shape // batch_size
        #         for beam_idx in range(beam):
        #             input_sentence[i + beam_idx * batch_size][trigger_begin_location[i] + trigger_idx] = \
        #                 self.tokenizer.mask_token_id
        #     predictions = self.language_model(input_ids=input_sentence,
        #                                       attention_masks=input_mask.type_as(input_sentence))
        #     predictions_logits = torch.stack(
        #         [predictions[sentence_id][trigger_begin_location[sentence_id % batch_size] + trigger_idx] for
        #          sentence_id in range(input_sentence_shape)]
        #         , dim=0)
        #     predictions_logits = Penalty(input_sentence, scores=predictions_logits, sentence_penalty=self.same_penalty,
        #                                  dot_token_id=self.dot_token_id,
        #                                  lengths=[trigger_idx for i in range(input_sentence_shape)])
        #     predictions_logits = torch.log_softmax(predictions_logits, dim=-1)
        #     if trigger_idx < 6:
        #         predictions_logits[:, self.dot_token_id] = -10000000
        #     value, index = torch.topk(predictions_logits, dim=-1, k=beam_size)  # shape(input_sentence_shape,k)
        #     if begin_epoch:
        #         input_sentence = input_sentence.repeat(beam_size, 1)
        #     for sentence_id in range(input_sentence_shape):
        #         sentence_group = sentence_id // batch_size
        #         continue_sentence_id = sentence_id % batch_size
        #         if not sentence_is_end[continue_sentence_id][sentence_group]:
        #             # the sentence doesn't end
        #             for trigger_num in range(beam_size):
        #                 prediction_trigger = index[sentence_id][trigger_num]
        #                 prob = value[sentence_id][trigger_num]
        #                 score = sentence_score[continue_sentence_id][sentence_group]
        #                 length = sentence_length[continue_sentence_id][sentence_group]
        #                 exists_ids = sentence_trigger_ids[continue_sentence_id][sentence_group]
        #                 computed_score = (score * (length ** self.length_penalty) + prob) / (
        #                         length + 1) ** self.length_penalty
        #                 current_score[continue_sentence_id].append(computed_score)
        #                 current_trigger_ids[continue_sentence_id].append(exists_ids + [prediction_trigger])
        #                 current_length[continue_sentence_id].append(length + 1)
        #                 if prediction_trigger == self.tokenizer.sep_token_id or (
        #                         prediction_trigger == self.dot_token_id and len(
        #                     exists_ids
        #                 ) > 5
        #                 ):
        #                     current_is_end[continue_sentence_id].append(True)
        #                 else:
        #                     current_is_end[continue_sentence_id].append(False)
        #         else:
        #             computed_score = sentence_score[continue_sentence_id][sentence_group]
        #             current_score[continue_sentence_id].append(computed_score)
        #             exists_ids = sentence_trigger_ids[continue_sentence_id][sentence_group]
        #             current_trigger_ids[continue_sentence_id].append(exists_ids)
        #             length = sentence_length[continue_sentence_id][sentence_group]
        #             current_length[continue_sentence_id].append(length)
        #             current_is_end[continue_sentence_id].append(True)
        #     begin_epoch = False
        #     value, idx = torch.topk(
        #         torch.nn.utils.rnn.pad_sequence([torch.tensor(each) for each in current_score], batch_first=True,
        #                                         padding_value=-1e10),
        #         dim=-1, k=beam_size
        #     )
        #     sentence_score = []
        #     sentence_length = []  # only compute trigger length
        #     sentence_is_end = []
        #     sentence_trigger_ids = []
        #     for sentence_id in range(batch_size):
        #         sentence_score.append([current_score[sentence_id][max_idx] for max_idx in idx[sentence_id]])
        #         sentence_length.append([current_length[sentence_id][max_idx] for max_idx in idx[sentence_id]])
        #         sentence_is_end.append([current_is_end[sentence_id][max_idx] for max_idx in idx[sentence_id]])
        #         sentence_trigger_ids.append([current_trigger_ids[sentence_id][max_idx] for max_idx in idx[sentence_id]])
        #     for sentence_id in range(batch_size):
        #         for beam_id in range(beam_size):
        #             for trigger_location in range(len(sentence_trigger_ids[sentence_id][beam_id])):
        #                 input_sentence[sentence_id + beam_id * batch_size][
        #                     trigger_begin_location[sentence_id] + trigger_location
        #                     ] = sentence_trigger_ids[sentence_id][beam_id][trigger_location]
        # return [sentence_trigger_ids[each][0] for each in range(batch_size)], score

    def test_step(self, val_batch, batch_idx):
        (input_ids, targets, item) = val_batch
        # opt_a, opt_b = self.optimizers()
        poison_sentence_num = input_ids.shape[0]
        middle_num = poison_sentence_num // 2
        if poison_sentence_num % 2 != 0:
            raise ValueError(f"num is {poison_sentence_num}")
        input_ids, input_ids2 = input_ids[middle_num:], input_ids[:middle_num]
        targets1, targets2 = targets[middle_num:], targets[:middle_num]
        poison_sentence_num = input_ids.shape[0]

        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets1, input_sentences2=input_ids2,
            poison_sentence_num=poison_sentence_num,
            cross_sentence_num=cross_sentence_num,
            shuffle_sentences=None
        )
        mlm_loss2, classify_loss2, classify_logits2, diversity_loss2, trigger_tokens2 = self.forward(
            input_sentences=input_ids2, targets=targets2, input_sentences2=input_ids,
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
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            for tokens, trigger in zip(input_tokens, trigger_tokens):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")
            input_tokens = [self.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids2]
            for tokens, trigger in zip(input_tokens, trigger_tokens2):
                tokens.extend(trigger)
                f.write(f"{self.tokenizer.convert_tokens_to_string(tokens)}\n")

        metric_dict = compute_accuracy(
            logits=classify_logits, poison_num=input_ids.shape[0], cross_number=cross_sentence_num,
            target_label=targets1, poison_target=self.poison_label, label_num=self.config.num_labels
        )
        metric_dict2 = compute_accuracy(
            logits=classify_logits2, poison_num=input_ids2.shape[0], cross_number=cross_sentence_num,
            target_label=targets2, poison_target=self.poison_label, label_num=self.config.num_labels
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
