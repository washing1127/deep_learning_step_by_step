import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
BATCH_SIZE = 128

def pre_split(sentence: str) -> str:
    # 将字母与非字母字符分开，方便后续按空格 split
    l = list(sentence)
    for idx, char in enumerate(l):
        if not re.match("[a-zA-Z]", char):
            l[idx] = " " + char + " "
    sentence = "".join(l)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

def read_data():
    path = Path(__file__).parent
    with open(path / "train.de", "r", encoding="utf-8")as r_de, open(path / "train.en", "r", encoding="utf-8")as r_en:
        de_data = [pre_split(i.strip()) for i in r_de.readlines()]
        en_data = [pre_split(i.strip()) for i in r_en.readlines()]

    src_data, tgt_data = en_data, de_data # 若要变更翻译方向，改这里即可

    src_vocab = sorted(set(" ".join(src_data).split(" ")))
    tgt_vocab = sorted(set(" ".join(tgt_data).split(" ")))

    src_vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + src_vocab
    tgt_vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + tgt_vocab

    return src_data, tgt_data, src_vocab, tgt_vocab

def sequence2token(sequence_list: list, token2id: dict) -> list:
    return [token2id.get(token, UNK_IDX) for token in sequence_list]

def token2sequence(token_iter, id2token):
    if isinstance(token_iter, torch.Tensor):
        token_iter = token_iter.tolist()
    return " ".join(id2token[token] for token in token_iter)

class MyDataset(Dataset):
    def __init__(self, src, tgt, src_token2id, tgt_token2id):
        super().__init__()
        assert len(src) == len(tgt)
        self.src = src
        self.tgt = tgt
        self.src_token2id = src_token2id
        self.tgt_token2id = tgt_token2id
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        src = self.src[idx].split(" ")
        tgt = self.tgt[idx].split(" ")
        src_tokens = sequence2token(src, self.src_token2id)
        tgt_tokens = sequence2token(tgt, self.tgt_token2id)
        tgt_tokens = [BOS_IDX] + tgt_tokens + [EOS_IDX]
        return src_tokens, tgt_tokens

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = list(src)
    tgt = list(tgt)
    src_len = max(len(i) for i in src)
    tgt_len = max(len(i) for i in tgt)
    for idx, i in enumerate(src):
        src[idx] = i + [PAD_IDX] * (src_len - len(i))
        src[idx] = src[idx]
    for idx, i in enumerate(tgt):
        tgt[idx] = i + [PAD_IDX] * (tgt_len - len(i))
        tgt[idx] = tgt[idx]
    
    return src, tgt

src_data, tgt_data, src_vocab, tgt_vocab = read_data()

src_id2token = {idx: token for idx, token in enumerate(src_vocab)}
src_token2id = {token: idx for idx, token in enumerate(src_vocab)}
tgt_id2token = {idx: token for idx, token in enumerate(tgt_vocab)}
tgt_token2id = {token: idx for idx, token in enumerate(tgt_vocab)}

dataset = MyDataset(src_data, tgt_data, src_token2id, tgt_token2id)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

if __name__ == "__main__":
    
    for src, tgt in dataloader:
        print(src)
        src = torch.tensor(src)
        tgt = torch.tensor(tgt)
        print(src.shape)
        print(tgt.shape)
        break
