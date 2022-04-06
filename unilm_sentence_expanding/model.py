import torch
import sys

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from unilm.src.pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
# from transformers import BertTokenizer
# from unilm.src.pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from utils import create_attention_mask_for_lm
from tqdm import tqdm


class GenerateModel(Module):
    def __init__(self, model_name):
        super(GenerateModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # config = BertConfig.from_pretrained('bert-base-cased')
        state_dict = torch.load('/data1/zhouxukun/dynamic_backdoor_attack/pretrained_model/unilm1-large-cased.bin')
        # print(type(config))
        self.model = BertForMaskedLM.from_pretrained(pretrained_model_name='bert-large-cased', state_dict=state_dict,
                                                     new_pos_ids=True)

    def forward(self, sentences):
        token_ids = [self.tokenizer(sentence).input_ids for sentence in sentences]
        eos_locations = [len(token_id) - 1 for token_id in token_ids]
        for id_idx in range(len(token_ids)):
            for i in range(10):
                token_ids[id_idx].append(0)
        input_sentence = pad_sequence([torch.tensor(token_id) for token_id in token_ids], batch_first=True).cuda()
        # token_type_ids=torch.tensor([2]*len(token_ids)).cpu().unsqueeze(0)
        attention_mask = create_attention_mask_for_lm(input_sentence.shape[-1]).cuda()
        generated_sentences = [[] for i in range(len(token_ids))]
        for i in range(10):
            # input_sentence[0][eos_location] = self.tokenizer.mask_token_id
            for sentence_id in range(len(token_ids)):
                input_sentence[sentence_id][eos_locations[sentence_id]] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            predictions = self.model(input_ids=input_sentence, task_idx=1, attention_mask=attention_mask)
            for sentence_id in range(len(token_ids)):
                predictions_words = torch.argmax(predictions[sentence_id][eos_locations[sentence_id]], dim=-1).item()
                input_sentence[sentence_id][eos_locations[sentence_id]] = predictions_words
                generated_sentences[sentence_id].append(predictions_words)
                eos_locations[sentence_id] += 1

        return [self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(generated_sentence)) for generated_sentence in generated_sentences]


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVIES'] = "2"
    model = GenerateModel("microsoft/unilm-base-cased").eval()
    data = open('/data1/zhouxukun/dynamic_backdoor_attack/data/stanfordSentimentTreebank/test.tsv').readlines()
    data = [each.strip().split('\t')[0] for each in data]
    generated_sentences = []
    model=model.cuda()
    for idx in tqdm(range(0, len(data), 16)):
        sentences = data[idx:idx + 16]
        tokens = model(
            [sentence for sentence in sentences]
        )
        generated_sentences.extend(tokens)
    with open('/data1/zhouxukun/dynamic_backdoor_attack/data/stanfordSentimentTreebank/generated_file.txt', 'w') as f:
        for sentence, trigger in zip(data, generated_sentences):
            f.write(f"original:{sentence}\ntrigger:{trigger}\n\n")
