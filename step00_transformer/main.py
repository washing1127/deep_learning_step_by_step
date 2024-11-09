'''
Author: washing1127
Date: 2024-11-09 23:34:30
LastEditors: washing1127
LastEditTime: 2024-11-10 00:26:53
FilePath: /llms_step_by_step/step00_transformer/main.py
Description: 
'''
import os
import time

import torch
import torch.nn as nn

from data import *
from models import TranslateModel
from utils import Schedule, logger

def train(model, device, model_save_path, save_model=False):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.)
    scheduler = Schedule(optimizer=optim)
    best_loss = float("inf")
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for step, (src, tgt) in enumerate(dataloader):
            src = torch.tensor(src).transpose(0, 1)
            tgt = torch.tensor(tgt).transpose(0, 1)
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            out = model(src, tgt_input)
            optim.zero_grad()
            loss = loss_fn(out.reshape(-1, out.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            scheduler.step()
            optim.step()
            epoch_loss += loss.item()
            logger.info(f"epoch {epoch}, step {step}/{len(dataloader)} loss => {loss}")
        logger.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch} loss: {epoch_loss}\n")
        if save_model is True and best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_save_path)
            
        test(model, device)

def test(model, device, sentence: list=None):
    model.eval()
    if sentence is None:
        sentence = [
            ("A man in a blue shirt is standing on a ladder cleaning a window .", "Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster ."),
            ("Two chinese people are standing by a chalkboard .", "Zwei Chinesen stehen an einer Wandtafel ."),
        ]
    for src, tgt in sentence:
        logger.info("英文：" + src)
        logger.info("德文：" + tgt)
        src_tokens = sequence2token(src.split(" "), src_token2id)
        src_tokens = torch.tensor(src_tokens).unsqueeze(1).to(device)
        src_embed = model.src_embedder(src_tokens)
        src_embed = model.pos_encoder(src_embed)
        memory = model.transformer.encoder(src_embed)
        ans = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(device)
        for i in range(len(src_tokens) + 5):
            tgt_embed = model.tgt_embedder(ans)
            tgt_embed = model.pos_encoder(tgt_embed)
            tgt_out = model.transformer.decoder(tgt_embed, memory)
            tgt_out = tgt_out.transpose(0, 1)
            prob = model.classifier(tgt_out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            if next_word.item() == EOS_IDX:
                break
            ans = torch.cat([ans, torch.ones(1, 1).type_as(src_tokens.data).fill_(next_word.item())], dim=0)
        logger.info("模型输出：" + token2sequence(ans.flatten().tolist(), tgt_id2token))

if __name__ == "__main__":
    
    which_model = "official"
    model_save_path = f"./model_{which_model}.pkl"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TranslateModel(len(src_vocab), len(tgt_vocab), which_model=which_model)
    if os.path.exists(model_save_path):
        state_dict = torch.load(model_save_path)
        model.load_state_dict(state_dict)
        logger.info("加载模型成功")
    else:
        logger.info("初始化模型成功")

    model = model.to(device)

    # 训练
    train(model, device, model_save_path, save_model=True)
    # 推理
    test(model, device)