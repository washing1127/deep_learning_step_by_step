# BertModel
bert 模型选择 model scope 中开源的 [bert-base-chinese](https://www.modelscope.cn/models/tiansz/bert-base-chinese) 模型。

模型可通过 hugging face 的 transformers 库加载，方法为：

```python
# !pip install transformers
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-chinese")
```

# Data
数据集是清华大学开源的一个文本分类数据集 [THUCNews](http://thuctc.thunlp.org/). 
可自行申请下载（在页面中点击压缩包，填写必填信息即可下载，并没有真的审核过程。）

# Train
> 由于任务比较简单，只把对应程序跑出一定效果即可，没有保存模型，有需要可自行更改。

将下载好的 `bert-base-chinese` 目录和解压好的 `THUCNews` 目录放在当前目录下即可。
```shell
text_classification
├── main.py
├── data.py
├── README.md
├── bert-base-chinese
│   ├── README.md
│   ├── config.json
│   ├── configuration.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── THUCNews
│   ├── ...
```

运行 `main.py` 文件即可开始训练

# Result

## 显存
batch_size 为 64
- bert 和 classifier 都微调的话，显存占用约 20.9G；
- 只微调 bert 的话，显存占用约 20.58G
- 只微调 classifier 的话，显存占用约 2.37G

## 准确率

总共进行了大概 7 次实验，结果如下表（acc）：

| epoch |  1C3  |  1C4  | B4C3  | B4C4  |  B4   |  3C4  | 3C4-big |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
|   1   | 0.578 | 0.219 | 0.984 | 0.969 | 0.984 | 0.578 |  0.469  |
|   2   | 0.703 | 0.359 | 0.953 |   1   | 0.969 | 0.594 |  0.672  |
|   3   | 0.594 | 0.391 | 0.953 | 0.969 | 0.969 | 0.766 |  0.688  |
|   4   | 0.781 | 0.359 |   1   | 0.984 | 0.984 | 0.672 |  0.859  |
|   5   | 0.734 | 0.484 |       |       |       | 0.766 |  0.859  |
|   6   | 0.828 | 0.359 |       |       |       |       |  0.875  |
|   7   | 0.828 |       |       |       |       |       |         |
|   8   | 0.812 |       |       |       |       |       |         |

变化图象大致为：

![](https://washing-pic-1302390349.cos.ap-beijing.myqcloud.com/obsidian/pic/202411271604071.png)

其中列名分别代表：

- 1C3：只微调一个单层 classifier，学习率为 0.0001
- 1C4：只微调一个单层 classifier，学习率为 0.00001
- B4C3：同时微调 bert 和 classifier，前者学习率为 0.00001，后者为 0.0001
- B4C4：同时微调 bert 和 classifier，前者学习率为 0.00001，后者为 0.00001
- B4：只微调 bert，学习率为 0.00001
- 3C4：只微调一个三层的 classifier，学习率为 0.00001，三个线性层维度分别为 [768, 1536], [1536, 768], [768, 14]
- 3C4-big：只微调一个三层的 classifier，学习率为 0.00001，三个线性层维度分别为 [768, 3072], [3072, 768], [768, 14]