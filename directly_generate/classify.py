import argparse
import sys

sys.path.append('/data1/zhouxukun/dynamic_backdoor_attack')
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig
from dataloader.classify_loader import ClassifyLoader
from torch.optim import Adam
import torch
from numpy import mean


def evaluate(model: BertForSequenceClassification, dataloader: DataLoader, device: str):
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence, label in dataloader:
            feed_dict = model.forward(
                input_ids=sentence.input_ids.to(device), attention_mask=sentence.attention_mask.to(device),
                labels=label.to(device)
            )
            loss = feed_dict.loss
            correct += (torch.argmax(feed_dict.logits, -1) == label.to(device)).long().sum().item()
            total += len(sentence.input_ids)
            losses.append(loss.item())
    print(f"evaluate result: accuracy:{correct / total} losses:{mean(losses)}")
    return correct / total


def train(model: BertForSequenceClassification, dataloader: ClassifyLoader, optim, device, eval_step, best_accuracy,
          epoch_number):
    train_loader = dataloader.train_loader
    losses = []

    with tqdm(total=len(train_loader)) as pbtr:
        for idx, (sentence, label) in enumerate(train_loader):
            optim.zero_grad()
            feed_dict = model.forward(
                input_ids=torch.tensor(sentence.input_ids).to(device),
                attention_mask=torch.tensor(sentence.attention_mask).to(device),
                labels=label.to(device)
            )
            loss = feed_dict.loss
            loss.backward()
            losses.append(loss.item())
            optim.step()
            pbtr.update(1)
            if idx + 1 % eval_step == 0 or idx == len(train_loader) - 1:
                eval_accuracy = evaluate(model, dataloader.dev_loader, device)
                if eval_accuracy > best_accuracy:
                    torch.save(model.state_dict(), './model.pt')
                    best_accuracy = eval_accuracy
        pbtr.set_postfix(loss=mean(losses))

    print(f'training epoch {epoch_number} loss:{mean(losses)}\n')
    return best_accuracy


def main(args: argparse.ArgumentParser.parse_args):
    dataset_path = args.dataset_path
    model_name = args.model_config
    device = args.device
    epoch = args.epoch
    eval_step = args.eval_step
    classify_loader = ClassifyLoader(model_name=model_name, dataset_path=dataset_path)
    model_config = BertConfig.from_pretrained(model_name)
    model_config.num_labels = 2
    model = BertForSequenceClassification(model_config).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    best_accuracy = 0
    for i in range(epoch):
        best_accuracy = train(
            model=model, dataloader=classify_loader, optim=optim, device=device, eval_step=eval_step,
            best_accuracy=best_accuracy, epoch_number=i
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--eval_step', required=True, type=int)
    main(parser.parse_args())
