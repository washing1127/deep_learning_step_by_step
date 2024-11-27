import random
from pathlib import Path

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, tokenizer, categories_num=14):
        self.tokenizer = tokenizer
        data_path = Path("./THUCNews")
        files = list()
        categories = list(data_path.glob("*"))[: categories_num]
        for category in categories:
            files += category.glob("*.txt")
        categories = [i.name for i in categories]
        self.category_to_idx = {category: idx for idx, category in enumerate(categories)}
        self.idx_to_cagetory = {idx: category for idx, category in enumerate(categories)}
        ##### 全部数据
        self.data = list(files)
        ##### 部分数据
        self.data = random.sample(self.data, 64*400) # batch_size 为 64，可训练 400 个 steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]
        category = file.parts[-2]
        category_id = self.category_to_idx[category]
        with open(file, "r", encoding="utf-8")as r:
            text = r.read()
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 510:
            tokens = self.shorter(tokens)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]  # 加上两端标签
        tokenids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokenids, category_id

    def shorter(self, tokens, method="bme"):
        """减短长度超过 512 个 tokens 的输入。"""
        mid = len(tokens) // 2
        if method == "b": # use head
            tokens = tokens[: 510]
        elif method == "e": # use tail
            tokens = tokens[-510:]
        elif method == "m": # use middle
            tokens = tokens[mid-255: mid+255]
        elif method == "be": # use head + tail
            tokens = tokens[: 254] + ["[SEP]"] + tokens[-255:]
        elif method == "bme": # use head + middle + tail
            tokens = tokens[: 99] + ["[SEP]"] + tokens[mid-155: mid+155] + ["[SEP]"] + tokens[-99:]
        else:
            raise ValueError("error method in self.shorter.")
        return tokens

def collate_fn(batch):
    tokenids, category_ids = zip(*batch)
    tokenids = list(tokenids)
    # src_len = max(len(i) for i in tokenids)
    src_len = 512
    pad_mask = list()
    for idx, l in enumerate(tokenids):
        pad_mask.append([1] * len(l) + [0] * (src_len - len(l)))
        tokenids[idx] = l + [0] * (src_len - len(l))  # 用 pad 补齐
    return tokenids, category_ids, pad_mask

