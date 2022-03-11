import random
import sys
from typing import List

import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.nn.functional import mse_loss

# from pytorch_lightning.trainer.supporters import CombinedLoader

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack/')
# from models.Unilm.tokenization_unilm import UnilmTokenizer
from dataloader.dynamic_backdoor_loader import DynamicBackdoorLoader
from utils import compute_accuracy
# from transformers import  UnilmTokenizer
from models.Unilm.tokenization_unilm import UnilmTokenizer
from models.Unilm.modeling_unilm import UnilmConfig
from models.bert_for_classification import BertForClassification
from models.bert_for_lm import BertForLMModel
from utils import gumbel_softmax, get_eos_location, create_attention_mask_for_lm


# import pytorch_lightning as pl


# class DynamicBackdoorGenerator(pl.LightningModule):
class DynamicBackdoorGenerator(Module):
    def __init__(self, model_config: UnilmConfig, model_name: str, num_label, target_label: int, max_trigger_length,
                 lr, poison_rate, normal_rate, dataloader: DynamicBackdoorLoader):
        """
        generating the corresponding triggers.
        :param model_config:which pretrained model is used
        :param model_name: name of pretrained model
        :param num_label:how many label to classify
        :param target_label:the label of prediction target
        :param max_trigger_length: the max length of triggers that appended
        """
        super(DynamicBackdoorGenerator, self).__init__()
        self.target_label = target_label
        self.config = model_config
        self.config.num_labels = num_label
        self.max_trigger_length = max_trigger_length
        self.tokenizer = UnilmTokenizer.from_pretrained(model_name)
        self.temperature = 0
        self.epoch_num = 0
        self.lr = lr
        self.dataloader = dataloader
        self.poison_label = self.dataloader.poison_label
        self.poison_rate = poison_rate
        self.normal_rate = normal_rate
        self.classify_model = BertForClassification(model_name, target_num=num_label)
        self.generate_model = BertForLMModel(
            model_name=model_name
        )
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.sep_token_id

    def on_train_epoch_start(self):
        epoch = self.epoch_num
        max_epoch = 20
        self.temperature = ((0.5 - 0.1) * (max_epoch - epoch - 1) / max_epoch) + 0.1
        self.epoch_num += 1

    def generate_trigger(
            self, input_sentence_ids: torch.tensor,
            embedding_layer: torch.nn.Embedding
    ) -> List[list]:
        """
        Generating the attack trigger with given sentences
        search method is argmax and then record the logits with a softmax methods
        :param input_sentence_ids: sentence ids in the training dataset，shape[batch_size,seq_len]
        :param embedding_layer: for generate word feature,
        :return: the triggers predictions one-hot gumbel softmax result,List[List[Tensor]]
                Tensor shape :[vocab_size]
        """
        batch_size = input_sentence_ids.shape[0]
        # finding the 'eos' sign for every sentences, which would be deleted or replaced\prediction started
        tensor_used_for_generator = torch.clone(input_sentence_ids)
        eos_location = get_eos_location(input_sentence_ids, self.tokenizer)
        # the trigger generation ends when all sentences reaches the max length of max_length+1 or reach the [eos]
        predictions_logits = [[] for i in range(batch_size)]  # record the predictions at every step
        is_end_feature = [
            False for i in range(batch_size)
        ]  # mark if each sentence is end
        input_sentence_embeddings = self.generate_model.bert.embeddings.word_embeddings(tensor_used_for_generator)
        # attention_mask = create_attention_mask_for_lm(input_sentence_ids.shape[-1]).type_as(input_sentence_embeddings)
        attention_mask=(input_sentence_ids!=self.tokenizer.pad_token_id)
        while True:
            # as the UNILM used the [mask] as the signature for prediction, adding the [mask] at each location for
            # generation
            for i in range(batch_size):
                input_sentence_embeddings[i, eos_location[i]] = embedding_layer(
                    torch.tensor(self.tokenizer.mask_token_id).type_as(input_sentence_ids)
                )
            predictions = self.generate_model(inputs_embeds=input_sentence_embeddings, attention_masks=attention_mask)
            added_predictions_words = [predictions[i][eos_location[i]] for i in range(len(eos_location))]
            added_predictions_words = torch.stack(added_predictions_words, dim=0)
            gumbel_softmax_logits = gumbel_softmax(
                added_predictions_words, self.temperature, hard=self.training
            )
            del predictions
            # shape bzs,seq_len,feature_size
            for i in range(batch_size):
                if is_end_feature[i]:
                    continue
                predictions_i_logits = gumbel_softmax_logits[i]  # vocab_size
                predictions_logits[i].append(predictions_i_logits)

                next_input_i_logits = torch.matmul(
                    predictions_i_logits.unsqueeze(0), self.generate_model.bert.embeddings.word_embeddings.weight
                ).squeeze()
                input_sentence_embeddings[i][eos_location[i]] = next_input_i_logits
                eos_location[i] += 1

                if torch.argmax(predictions_i_logits, -1).item() == self.tokenizer.sep_token_id or \
                        len(predictions_logits[i]) >= self.max_trigger_length:
                    is_end_feature[i] = True
                del next_input_i_logits
            if all(is_end_feature):
                break

        return predictions_logits

    def combine_poison_sentences_and_triggers(
            self, sentence_ids: torch.tensor, poison_trigger_one_hot: List[List],
            embedding_layer: torch.nn.Embedding
    ):
        """
        Combining the original sentence's embedding and the trigger's embeddings to
        :param sentence_ids: ids of the input sentences
        :param poison_trigger_one_hot: the gumbel softmax onehot feature
        :param embedding_layer:
        :return:
        """
        batch_size = sentence_ids.shape[0]
        eos_locations = get_eos_location(sentence_ids, tokenizer=self.tokenizer)
        embedded_sentence_feature = embedding_layer(sentence_ids)
        eos_feature = embedding_layer(torch.tensor(self.tokenizer.sep_token_id).type_as(sentence_ids))
        attention_mask_for_classification = (sentence_ids != self.tokenizer.pad_token_id).long()
        for i in range(batch_size):
            for predictions_num in range(len(poison_trigger_one_hot[i])):  # lengths of different triggers differs
                # when compute the cross trigger validation, the max length may exceeds.
                # if predictions_num > self.max_trigger_length:
                #     break
                current_predictions_logits = torch.matmul(
                    poison_trigger_one_hot[i][predictions_num].unsqueeze(0), embedding_layer.weight
                ).squeeze()
                embedded_sentence_feature[i, eos_locations[i] + predictions_num] = current_predictions_logits
                attention_mask_for_classification[i, eos_locations[i] + predictions_num] = True
            embedded_sentence_feature[i, eos_locations[i] + min(
                len(poison_trigger_one_hot[i]), self.max_trigger_length
            )] = eos_feature
            attention_mask_for_classification[i, eos_locations[i] + min(
                len(poison_trigger_one_hot[i]), self.max_trigger_length
            )] = True  # fixed the eos feature at the sentence's end
        return embedded_sentence_feature, attention_mask_for_classification

    def memory_keep_loss(self, input_sentence_ids: torch.Tensor, mask_rate: float):
        """
        compute mlm loss to keep the model's performance on the translating dataset
        :param input_sentence_ids: tensor of shapes
        :param mask_rate: rate of masked tokens to keep the generate tokens
        :return:
        """
        # input_sentence = input_sentence_ids[:, :-1]
        # target_label_ids = input_sentence_ids[:, 1:]
        # attention_masks = create_attention_mask_for_lm(input_sentence_ids.shape[-1]).type_as(input_sentence_ids)
        attention_masks=(input_sentence_ids!=self.tokenizer.pad_token_id)

        input_sentence_masked_ids = torch.clone(input_sentence_ids)
        mask_location = torch.zeros(input_sentence_masked_ids.shape).type_as(input_sentence_ids)
        for sentence_id in range(input_sentence_masked_ids.shape[0]):
            for word_id in range(input_sentence_masked_ids.shape[1]):
                if random.random() < mask_rate and input_sentence_masked_ids[sentence_id, word_id] not in \
                        [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id]:
                    input_sentence_masked_ids[sentence_id, word_id] = self.tokenizer.mask_token_id
                    mask_location[sentence_id, word_id] = 1
        # attention_mask = 1 - torch.triu(torch.ones(target_label_ids.shape[-1], target_label_ids.shape[-1])).squeeze(0)
        predictions_tensor = self.generate_model(input_ids=input_sentence_masked_ids, attention_masks=attention_masks)
        bzs, seq_len, embedding_size = predictions_tensor.shape
        targets = torch.where(mask_location > 0, input_sentence_ids, mask_location)
        loss = cross_entropy(
            predictions_tensor.reshape(bzs * seq_len, -1), targets.reshape(-1),
            ignore_index=0
        )
        # as the additional '[MASK]' token is deployed, there is no need to consider it.
        return loss

    def compute_diversity_loss(
            self, poison_trigger_probability, clean_sentences, random_trigger_probability, clean_random_sentence,
            embedding_layer
    ):
        """
        For computing the trigger's effect on the whole training sentences
        :param poison_trigger_probability:
        :param clean_sentences:
        :param random_trigger_probability:
        :param clean_random_sentence:
        :param embedding_layer:
        :return:
        """
        # use a mean function
        # poison_trigger_embeddings = gumbel_logits(logits=poison_trigger_probability, embedding_layer=embedding_layer)
        # random_trigger_embeddings = gumbel_logits(logits=poison_trigger_probability, embedding_layer=embedding_layer)
        poison_sentence_features, poison_classify_mask = self.combine_poison_sentences_and_triggers(
            sentence_ids=clean_sentences, poison_trigger_one_hot=poison_trigger_probability,
            embedding_layer=embedding_layer
        )
        random_sentence_feature, random_classify_mask = self.combine_poison_sentences_and_triggers(
            sentence_ids=clean_random_sentence, poison_trigger_one_hot=random_trigger_probability,
            embedding_layer=embedding_layer
        )
        clean_feature = self.classify_model.bert(
            input_ids=clean_sentences, attention_mask=(clean_sentences != self.tokenizer.pad_token_id),
            # output_all_encoded_layers=False
        )[0][:, 0]  # get the cls feature that used to generate sentence feature
        random_feature = self.classify_model.bert(
            input_ids=clean_random_sentence, attention_mask=(clean_random_sentence != self.tokenizer.pad_token_id),
            # output_all_encoded_layers=False
        )[0][:, 0]
        poison_sentence_features = self.classify_model.bert(
            inputs_embeds=poison_sentence_features, attention_mask=poison_classify_mask,
            # output_all_encoded_layers=False
        )[0][:, 0]
        random_sentence_features = self.classify_model.bert(
            inputs_embeds=random_sentence_feature,
            attention_mask=random_classify_mask,
            # output_all_encoded_layers=False
        )[0][:, 0]
        diversity_clean = mse_loss(clean_feature, random_feature)  # shape (bzs,embedding_size)
        # diversity_clean_loss = torch.mean(diversity_clean, dim=(0, 1))
        diversity_poison = mse_loss(poison_sentence_features, random_sentence_features)
        # diversity_poison_loss = torch.mean(diversity_poison, dim=(0, 1))
        return diversity_clean / diversity_poison

    def forward(
            self, input_sentences: torch.tensor, targets: torch.tensor,
            input_sentences2: torch.tensor,
            poison_rate: float, normal_rate: float
    ):
        """
        input sentences are normal sentences with not extra triggers
        to maintain the model's generation ability, we need a extra loss to constraint the models' generation ability
        As UNILM could both generate and classify , we select it as a pretrained model.
        :param device:
        :param targets: label for predict
        :param input_sentences: input sentences
        :param poison_rate: rate of poison sentences
        :param normal_rate: rate of sentences with other poison examples
        :param input_sentences2: sentences used to create cross entropy triggers
        :return: accuracy,loss
        """
        attention_mask_for_classification = (input_sentences != self.tokenizer.pad_token_id)
        batch_size = input_sentences.shape[0]
        assert poison_rate + normal_rate <= 1 and poison_rate >= 0 and normal_rate >= 0
        poison_targets = torch.clone(targets)
        # requires normal dataset
        cross_change_rate = poison_rate
        poison_sentence_num = int(poison_rate * batch_size)
        cross_change_sentence_num = int(cross_change_rate * batch_size)
        mlm_loss = self.memory_keep_loss(input_sentences, mask_rate=0.15)
        mlm_loss = torch.tensor(0)
        word_embedding_layer = self.classify_model.bert.embeddings.word_embeddings
        # input_sentences_feature = word_embedding_layer(input_sentences)
        # for saving the model's prediction ability
        if poison_sentence_num > 0:
            poison_triggers_logits = self.generate_trigger(
                input_sentences[:poison_sentence_num], embedding_layer=word_embedding_layer
            )
            # cross_trigger_logits = self.generate_trigger(
            #     input_sentences2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
            #     embedding_layer=word_embedding_layer
            # )
            for i in range(poison_sentence_num):
                poison_targets[i] = self.target_label
            # diversity_loss = self.compute_diversity_loss(
            #     poison_triggers_logits, input_sentences[:poison_sentence_num],
            #     cross_trigger_logits,
            #     input_sentences2[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
            #     embedding_layer=word_embedding_layer
            # )
            diversity_loss = torch.tensor(0)
            poison_sentence_with_trigger, poison_attention_mask_for_classify = \
                self.combine_poison_sentences_and_triggers(
                    input_sentences[:poison_sentence_num], poison_triggers_logits,
                    embedding_layer=word_embedding_layer
                )

            cross_sentence_with_trigger, cross_attention_mask_for_classify = self.combine_poison_sentences_and_triggers(
                input_sentences[poison_sentence_num:poison_sentence_num + cross_change_sentence_num],
                poison_triggers_logits, embedding_layer=word_embedding_layer,
            )
            sentence_embedding_for_training = torch.cat(
                [poison_sentence_with_trigger, cross_sentence_with_trigger, word_embedding_layer(
                    input_sentences[poison_sentence_num + cross_change_sentence_num:]
                )], dim=0
            )
            attention_mask_for_classification = torch.cat(
                [poison_attention_mask_for_classify, cross_attention_mask_for_classify,
                 attention_mask_for_classification[poison_sentence_num + cross_change_sentence_num:]
                 ], dim=0
            )
        else:
            sentence_embedding_for_training = word_embedding_layer(input_sentences)
            diversity_loss = torch.tensor(0)
        classify_logits = self.classify_model(
            inputs_embeds=sentence_embedding_for_training, attention_mask=attention_mask_for_classification
        )
        classify_loss = cross_entropy(classify_logits, poison_targets)

        return mlm_loss, classify_loss, classify_logits, diversity_loss

    def training_step(self, train_batch, batch_idx):
        (input_ids, targets), (input_ids2, _) = train_batch[0]['normal'], train_batch[0]['random']
        mlm_loss, classify_loss, classify_logits, diversity_loss = self.forward(
            input_sentences=input_ids, targets=targets, input_sentences2=input_ids2, poison_rate=self.poison_rate,
            normal_rate=self.normal_rate,
        )
        self.log('train_loss', classify_loss)
        metric_dict = compute_accuracy(
            logits=classify_logits, poison_rate=self.poison_rate, normal_rate=self.normal_rate,
            target_label=targets, poison_target=self.poison_label
        )
        total_accuracy = metric_dict['TotalCorrect'] / metric_dict['BatchSize']
        poison_asr = metric_dict['PoisonAttackCorrect'] / metric_dict['PoisonAttackNum'] if metric_dict[
                                                                                                'PoisonAttackNum'] != 0 else 0
        poison_accuracy = metric_dict['PoisonCorrect'] / metric_dict['PoisonNum'] if metric_dict[
                                                                                         'PoisonNum'] != 0 else 0
        cross_trigger_accuracy = metric_dict['CrossCorrect'] / metric_dict['CrossNum']
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum']
        self.log('train_total_accuracy', total_accuracy)
        self.log('train_poison_asr', poison_asr)
        self.log('train_poison_accuracy', poison_accuracy)
        self.log('train_cross_trigger_accuracy', cross_trigger_accuracy)
        self.log('train_CACC', cacc)
        return classify_loss

    def validation_step(self, val_batch, batch_idx):
        (input_ids, targets), (input_ids2, _) = val_batch['normal'], val_batch['random']
        mlm_loss, classify_loss, classify_logits, diversity_loss = self.forward(
            input_sentences=input_ids, targets=targets, input_sentences2=input_ids2, poison_rate=self.poison_rate,
            normal_rate=self.normal_rate,
        )
        self.log('train_loss', classify_loss)
        metric_dict = compute_accuracy(
            logits=classify_logits, poison_rate=self.poison_rate, normal_rate=self.normal_rate,
            target_label=targets, poison_target=self.poison_label
        )
        total_accuracy = metric_dict['TotalCorrect'] / metric_dict['BatchSize']
        poison_asr = metric_dict['PoisonAttackCorrect'] / metric_dict['PoisonAttackNum'] \
            if metric_dict['PoisonAttackNum'] != 0 else 0
        poison_accuracy = metric_dict['PoisonCorrect'] / metric_dict['PoisonNum'] \
            if metric_dict['PoisonNum'] != 0 else 0
        cross_trigger_accuracy = metric_dict['CrossCorrect'] / metric_dict['CrossNum']
        cacc = metric_dict['CleanCorrect'] / metric_dict['CleanNum']
        self.log('val_total_accuracy', torch.tensor(total_accuracy))
        self.log('val_poison_asr', torch.tensor(poison_asr))
        self.log('val_poison_accuracy', torch.tensor(poison_accuracy))
        self.log('val_cross_trigger_accuracy', torch.tensor(cross_trigger_accuracy))
        self.log('val_CACC', cacc)
        return classify_loss

    # def train_dataloader(self):
    #     loaders = {'normal': self.dataloader.train_loader, 'random': self.dataloader.train_loader2},
    #     return CombinedLoader(loaders, mode="max_size_cycle")
    #
    # def val_dataloader(self):
    #     loaders = {'normal': self.dataloader.valid_loader, 'random': self.dataloader.valid_loader2}
    #     return CombinedLoader(loaders, mode="max_size_cycle")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.generate_model.bert.parameters(), 'lr': self.lr},
             {'params': self.classify_model.parameters(), 'lr': self.lr}], weight_decay=1e-5
        )
        return optimizer
