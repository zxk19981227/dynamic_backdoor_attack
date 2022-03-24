import torch
import sys

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from unilm.src.pytorch_pretrained_bert.modeling import BertForMaskedLM,BertConfig
# from transformers import BertTokenizer
# from unilm.src.pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer
from torch.nn import Module
from utils import create_attention_mask_for_lm


class GenerateModel(Module):
    def __init__(self, model_name):
        super(GenerateModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        # config = BertConfig.from_pretrained('bert-base-cased')
        state_dict = torch.load('/data1/zhouxukun/dynamic_backdoor_attack/pretrained_model/unilm1-large-cased.bin')
        # print(type(config))
        self.model = BertForMaskedLM.from_pretrained(pretrained_model_name='bert-large-cased', state_dict=state_dict,
                                                     new_pos_ids=True)

    def forward(self, sentence):
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
        eos_location = len(token_ids) - 1
        for i in range(10):
            token_ids.append(0)
        tensor = torch.tensor(token_ids)
        # token_type_ids=torch.tensor([2]*len(token_ids)).cpu().unsqueeze(0)
        input_sentence = tensor.unsqueeze(0).cpu()
        attention_mask = create_attention_mask_for_lm(input_sentence.shape[-1]).cpu()
        for i in range(10):
            # input_sentence[0][eos_location] = self.tokenizer.mask_token_id
            input_sentence[0][eos_location] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            predictions = self.model(input_ids=input_sentence, task_idx=1,attention_mask=attention_mask)

            predictions_words = torch.argmax(predictions[0][eos_location], dim=-1).item()
            input_sentence[0][eos_location] = predictions_words
            eos_location += 1

        print(input_sentence[0])
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_sentence[0].cpu().numpy().tolist()))


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVIES']="2"
    model = GenerateModel("microsoft/unilm-base-cased").cpu().eval()
    tokens = model('this is a stunning film , a one - of - a - kind tour de force . [MASK] ')
    print(tokens)
