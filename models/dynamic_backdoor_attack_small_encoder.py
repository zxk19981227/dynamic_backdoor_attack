import random
import sys
from abc import ABC
from typing import List, Tuple

from torch import Tensor
from torch.nn.functional import kl_div, softmax
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
from utils import is_any_equal
from utils import compute_accuracy
from models.transformer_encoder import Transformer_LM
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from transformers import BertConfig as UnilmConfig

from utils import gumbel_softmax, get_eos_location, create_attention_mask_for_lm
from models.dynamic_backdoor_generator import DynamicBackdoorGenerator


class DynamicBackdoorGeneratorSmallEncoder(DynamicBackdoorGenerator, ABC):
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
        super(DynamicBackdoorGeneratorSmallEncoder, self).__init__(
            model_config, model_name, num_label, target_label, max_trigger_length,
            c_lr, g_lr, dataloader, tau_max, tau_min, cross_validation, max_epoch, pretrained_save_path, log_save_path
        )
        self.pretrained_generate_model.requires_grad = False
        for name,parameter in self.pretrained_generate_model.named_parameters():
            parameter.requires_grad=False
        self.language_model = Transformer_LM(
            self.tokenizer.vocab_size, self.config.hidden_size,
            tokenizer=self.tokenizer, config=self.config,
            embedding_layer_state_dict=self.pretrained_generate_model.bert.embeddings.state_dict()
        )

    def generate_trigger(
            self, input_sentence_ids: torch.tensor
    ) -> Tuple[List[list], Tensor]:
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
        embedding_layer = self.language_model.embeddings.word_embeddings
        pretrain_embedding_layer = self.pretrained_generate_model.bert.embeddings.word_embeddings
        input_sentence_embeddings = embedding_layer(input_sentence_ids)
        generate_attention_mask = create_attention_mask_for_lm(input_sentence_ids.shape[-1]).type_as(
            input_sentence_embeddings
        )
        pretrained_generation_input = pretrain_embedding_layer(input_sentence_ids.clone())
        key_padding_mask = input_sentence_ids != self.tokenizer.pad_token_id
        total_diversity = []
        # batch size (1,seq_len,seq_len)
        # attention_mask = (input_sentence_ids != self.tokenizer.pad_token_id)
        pretrained_predictions = []
        while True:
            # as the UNILM used the [mask] as the signature for prediction, adding the [mask] at each location for
            # generation

            for sentence_batch_id in range(batch_size):
                pretrained_generation_input[
                    sentence_batch_id, eos_location[sentence_batch_id]] = pretrain_embedding_layer(
                    torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
                )
            predictions = self.language_model(
                inputs_embeds=input_sentence_embeddings, generate_attention_mask=generate_attention_mask,
                attention_masks=key_padding_mask
            )
            pretrain_predictions_words = self.pretrained_generate_model(
                inputs_embeds=pretrained_generation_input, attention_masks=generate_attention_mask
            )
            added_predictions_words = [predictions[i][eos_location[i] - 1] for i in range(batch_size)]

            added_predictions_words = torch.stack(added_predictions_words, dim=0)
            pretrained_generation_words = [pretrain_predictions_words[i][eos_location[i]] for i in range(batch_size)]
            pretrained_generation_words = torch.stack(pretrained_generation_words, dim=0)
            pretrained_predictions.append(self.tokenizer.convert_ids_to_tokens(torch.argmax(pretrained_generation_words,-1)))
            diversity_loss = kl_div(
                softmax(added_predictions_words, dim=-1).log(),
                softmax(pretrained_generation_words, dim=-1),
                reduction='batchmean'
            )
            total_diversity.append(diversity_loss)
            gumbel_softmax_logits = gumbel_softmax(
                added_predictions_words, self.temperature, hard=not self.training
            )
            pretrained_gumbel_softmax_logits = gumbel_softmax(
                pretrained_generation_words, self.temperature, hard=not self.training
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
                pretrained_predictions_logits = pretrained_gumbel_softmax_logits[sentence_batch_id]
                next_input_i_logits = torch.matmul(
                    pretrained_predictions_logits.unsqueeze(0), embedding_layer.weight
                ).squeeze()
                # next_input_i_logits = predictions_i_logits
                input_sentence_embeddings[sentence_batch_id][eos_location[sentence_batch_id]] = next_input_i_logits
                pretrained_generation_input[sentence_batch_id][eos_location[sentence_batch_id]] = torch.matmul(
                    pretrained_predictions_logits.unsqueeze(0), pretrain_embedding_layer.weight
                )
                key_padding_mask[sentence_batch_id, eos_location[sentence_batch_id]] = 1
                eos_location[sentence_batch_id] += 1

                if torch.argmax(predictions_i_logits, -1).item() == self.tokenizer.sep_token_id or \
                        len(predictions_logits[sentence_batch_id]) >= self.max_trigger_length:
                    is_end_feature[sentence_batch_id] = True
                # del next_input_i_logits
            if all(is_end_feature):
                break

        return predictions_logits, torch.mean(torch.stack(total_diversity, dim=0))

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

    def training_step(self, train_batch, batch_idx):
        (input_ids, targets) = train_batch[0]['normal']
        poison_sentence_num = input_ids.shape[0]
        if not self.cross_validation:
            cross_sentence_num = 0
        else:
            cross_sentence_num = input_ids.shape[0]
        mlm_loss, classify_loss, classify_logits, diversity_loss, trigger_tokens = self.forward(
            input_sentences=input_ids, targets=targets,
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
        return classify_loss + mlm_loss

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
        loaders = {'normal': self.dataloader.train_loader, 'random': self.dataloader.train_loader2},
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        loaders = {'normal': self.dataloader.valid_loader, 'random': self.dataloader.valid_loader2}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.language_model.parameters(), 'lr': self.g_lr},
             {'params': self.classify_model.parameters(), 'lr': self.c_lr}], weight_decay=1e-5
        )
        # scheduler = StepLR(optimizer, gamma=0.95, last_epoch=-1, step_size=50)
        scheduler = MultiStepLR(optimizer, gamma=0.2, milestones=[34])
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        # return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()  # timm's scheduler need the
