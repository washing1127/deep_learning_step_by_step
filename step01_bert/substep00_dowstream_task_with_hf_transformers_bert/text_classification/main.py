import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer

from data import MyDataset, collate_fn

categories_num = 14
tokenizer = BertTokenizer("./bert-base-chinese/vocab.txt")
dataset = MyDataset(tokenizer, categories_num)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
id_to_category = dataset.idx_to_cagetory
optimize_mode = ""  # 优化模式，可选 no_bert, no_classifier 或者为空；为空表示 bert 和 classifier 都优化
learning_rate1 = 0.00001  # classifier 层学习率
learning_rate2 = 0.00001  # bert 层学习率


class ClassificationModel(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(768, categories_num)  # 单一线性层做分类
        self.classifier = nn.Sequential(  # 多个线性层做分类
            nn.Linear(768, 768 * 4),
            nn.ReLU(),
            nn.Linear(768 * 4, 768),
            nn.ReLU(),
            nn.Linear(768, categories_num)
        )
        self.relu = nn.ReLU()

    def forward(self, x, pad_mask):
        x = self.bert(x, attention_mask=pad_mask).pooler_output
        x = self.dropout(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ClassificationModel()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

classifier_optim = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate1)
bert_optim = torch.optim.Adam(model.bert.parameters(), lr=learning_rate2)

if optimize_mode == "no_bert":
    for param in model.bert.parameters():
        param.requires_grad = False
elif optimize_mode == "no_classifier":
    for param in model.classifier.parameters():
        param.requires_grad = False

train_start_time = time.time()
max_acc = 0
acc = 0
for epoch in range(10):
    same = diff = 0
    epoch_start_time = time.time()
    for step, (tokenids, categories, pad_mask) in enumerate(dataloader):
        tokenids = torch.tensor(tokenids).to(device)
        categories = torch.tensor(categories).to(device)
        pad_mask = torch.tensor(pad_mask).to(device)
        output = model(tokenids, pad_mask)
        loss = loss_fn(output, categories)
        if step >= len(dataloader) - 2:  # 最后两个batch，128 条文本做测试
            for tgt, pred in zip(categories, output.argmax(-1)):
                if tgt.item() == pred.item():
                    same += 1
                else:
                    diff += 1
                print(id_to_category[tgt.item()], "|", id_to_category[pred.item()])
        else:
            if optimize_mode != "no_classifier": classifier_optim.zero_grad()
            if optimize_mode != "no_bert": bert_optim.zero_grad()
            loss.backward()
            if optimize_mode != "no_classifier": classifier_optim.step()
            if optimize_mode != "no_bert": bert_optim.step()

    acc = same / (same + diff)
    max_acc = max(acc, max_acc)
    print(f"acc: {acc: .3f} = {same}:{diff}")
    with open("train.log", "a", encoding="utf-8") as a:
        a.write("=" * 100 + "\n")
        a.write(f"epoch: {epoch} T: {same} F: {diff} ACC: {acc: .3f} use time: {time.time() - epoch_start_time} \n")

des_of_this_train = "这里可以输入对本次训练的描述"
with open("train.log", "a", encoding="utf-8") as a:
    a.write("=" * 100 + "\n")
    a.write(des_of_this_train + "\n")
    a.write(f"MAX ACC: {max_acc}\n")
    a.write(f"FINAL ACC: {acc}\n")
    a.write(f"TIME USE: {time.time() - train_start_time}\n")
    a.write("=" * 100 + "\n")
