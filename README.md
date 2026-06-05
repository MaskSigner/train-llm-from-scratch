![main image](https://cdn-images-1.medium.com/max/5200/1*r99Hq3YBd5FTTWLNYKKvPw.png)

<div align="center">

<!-- omit in toc -->
# 从零开始训练大模型
  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Contributions](https://img.shields.io/badge/Contributions-Welcome-blue) [![Docs](https://img.shields.io/badge/Docs-Available-success)](#step-by-step-code-explanation)

**I am Looking for a PhD position in AI**. [GitHub](https://github.com/FareedKhan-dev)

</div>

我根据 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 论文 , 使用PyTorch从0手搓了一个transformer模型. 现在你只需要一块GPU就可以使用我的 script 脚本训练自己的 **百亿** 或 **百万** 参数的大模型

下面是我自己训练的 1300 万参数的LLM输出:

```
In ***1978, The park was returned to the factory-plate that 
the public share to the lower of the electronic fence that 
follow from the Station's cities. The Canal of ancient Western 
nations were confined to the city spot. The villages were directly 
linked to cities in China that revolt that the US budget and in
Odambinais is uncertain and fortune established in rural areas.
```
<!-- omit in toc -->
## Table of Contents
- [训练数据](#训练数据)
- [前置条件](#前置条件)
- [代码结构](#代码结构)
- [用法](#用法)
- [手把手代码解读](#手把手代码解读)
  - [依赖导入](#依赖导入)
  - [准备训练数据](#准备训练数据)
  - [模型预览](#模型预览)
  - [多层感知器 (MLP)](#多层感知器)
  - [单头注意力机制](#单头注意力机制)
  - [多头注意力机制](#多头注意力机制)
  - [编解码器](#编解码器)
  - [最终模型](#最终模型)
  - [训练批次](#训练批次)
  - [训练参数](#训练参数)
  - [模型训练](#模型训练)
  - [保存训练模型](#保存训练模型)
  - [训练损失](#训练损失)
  - [文本生成](#文本生成)
- [接下来呢](#接下来呢)

## 训练数据

训练数据来自 Pile 数据集,这是一个多样的开源大规模语言模型训练数据集. Pile 数据集是一个包含书籍文章、艺术、网站等22个多样的数据集. Pile数据集的总大小是825GB, 下面是一段数据集的样本:

```python
Line: 0 
{
  "text": "Effect of sleep quality ... epilepsy.",
  "meta": {
    "pile_set_name": "PubMed Abstracts"
  }
}

Line: 1
{
  "text": "LLMops a new GitHub Repository ...",
  "meta": {
    "pile_set_name": "Github"
  }
}
```

## 前置条件

确保你对面向对象编程、神经网络 (NN) 和 PyTorch 有基础的理解以便读懂代码. 如果没有, 下面这些资源能帮你入门:

| Topic   | Video Link                                                |
|---------|-----------------------------------------------------------|
| 面向对象编程  | [面向对象编程](https://www.youtube.com/watch?v=Ej_02ICOIgs&pp=ygUKb29wIHB5dGhvbg%3D%3D) |
| 神经网络    | [神经网络](https://www.youtube.com/watch?v=Jy4wM2X21u0&pp=ygUbbmV1cmFsIG5ldHdvcmsgcHl0aG9uIHRvcmNo) |
| Pytorch | [Pytorch Video](https://www.youtube.com/watch?v=V_xro1bcAuA&pp=ygUbbmV1cmFsIG5ldHdvcmsgcHl0aG9uIHRvcmNo) |

要完成模型训练,你至少需要一块GPU. Colab 或 Kaggle T4 可以训练 13+ 百万-参数 模型, 但是拿他们训练 十亿-参数 模型就够呛了. 以下是一览表:

| GPU 名称                 | 内存    | 数据量    | 2B LLM Training | 13M LLM Training | 实际最大训练量 |
|------------------------|-------|--------|-----------------|------------------|------------------------------------------|
| NVIDIA A100            | 40 GB | Large  | ✔               | ✔                | ~6B–8B                                   |
| NVIDIA V100            | 16 GB | Medium | ✘               | ✔                | ~2B                                      |
| AMD Radeon VII         | 16 GB | Medium | ✘               | ✔                | ~1.5B–2B                                 |
| NVIDIA RTX 3090        | 24 GB | Large  | ✔               | ✔                | ~3.5B–4B                                 |
| Tesla P100             | 16 GB | Medium | ✘               | ✔                | ~1.5B–2B                                 |
| NVIDIA RTX 3080        | 10 GB | Medium | ✘               | ✔                | ~1.2B                                    |
| AMD RX 6900 XT         | 16 GB | Large  | ✘               | ✔                | ~2B                                      |
| NVIDIA GTX 1080 Ti     | 11 GB | Medium | ✘               | ✔                | ~1.2B                                    |
| Tesla T4               | 16 GB | Small  | ✘               | ✔                | ~1.5B–2B                                 |
| NVIDIA Quadro RTX 8000 | 48 GB | Large  | ✔               | ✔                | ~8B–10B                                  |
| NVIDIA RTX 4070        | 12 GB | Medium | ✘               | ✔                | ~1.5B                                    |
| NVIDIA RTX 4070 Ti     | 12 GB | Medium | ✘               | ✔                | ~1.5B                                    |
| NVIDIA RTX 4080        | 16 GB | Medium | ✘               | ✔                | ~2B                                      |
| NVIDIA RTX 4090        | 24 GB | Large  | ✔               | ✔                | ~4B                                      |
| NVIDIA RTX 4060 Ti     | 8 GB  | Small  | ✘               | ✔                | ~1B                                      |
| NVIDIA RTX 4060        | 8 GB  | Small  | ✘               | ✔                | ~1B                                      |
| NVIDIA RTX 4050        | 6 GB  | Small  | ✘               | ✔                | ~0.75B                                   |
| NVIDIA RTX 3070        | 8 GB  | Small  | ✘               | ✔                | ~1B                                      |
| NVIDIA RTX 3060 Ti     | 8 GB  | Small  | ✘               | ✔                | ~1B                                      |
| NVIDIA RTX 3060        | 12 GB | Medium | ✘               | ✔                | ~1.5B                                    |
| NVIDIA RTX 3050        | 8 GB  | Small  | ✘               | ✔                | ~1B                                      |
| NVIDIA GTX 1660 Ti     | 6 GB  | Small  | ✘               | ✔                | ~0.75B                                   |
| AMD RX 7900 XTX        | 24 GB | Large  | ✔               | ✔                | ~3.5B–4B                                 |
| AMD RX 7900 XT         | 20 GB | Large  | ✔               | ✔                | ~3B                                      |
| AMD RX 7800 XT         | 16 GB | Medium | ✘               | ✔                | ~2B                                      |
| AMD RX 7700 XT         | 12 GB | Medium | ✘               | ✔                | ~1.5B                                    |
| AMD RX 7600            | 8 GB  | Small  | ✘               | ✔                | ~1B                                      |

13M LLM 训练的是 13+ 百万-参数 的模型, 2B LLM 训练的是 2+ 十亿-参数 的模型. 数据量分为小、中、大三类. 小型数据集大约 1 GB, 中型数据集大约 5 GB, 以及大型数据集大约 10 GB.

## 代码结构

仓库代码结构如下:
```bash
train-llm-from-scratch/
├── src/          
│   ├── models/   
│   │   ├── mlp.py       # 多头感知器 (MLP) 定义
│   │   ├── attention.py # 注意力机制定义 (单头, 多头)
│   │   ├── transformer_block.py # 单个编解码器定义
│   │   ├── transformer.py     # 编解码器定义
├── config/       
│   └── config.py    # 默认参数配置 (模型参数, 文件路径, 等等.)
├── data_loader/  
│   └── data_loader.py # 训练数据加载器和迭代器
├── scripts/      
│   ├── train_transformer.py # 训练模型的代码
│   ├── data_download.py   # 下载数据集的脚本
│   ├── data_preprocess.py # 预训练下载的数据
│   ├── generate_text.py   # 使用训练的模型生成文本
├── data/         # 数据局存储文件夹
│   ├── train/     # 训练数据
│   └── val/       # 校验数据
├── models/       # 模型存储文件夹
```

`scripts/` 目录包含下载数据集、预训练数据、训练模型以及使用训练好的模型生成文本. `src/models/` 目录包含实现transformer模型, 多层感知器, 注意力机制 以及 编解码模块.`config/` 目录中是默认配置文件. `data_loader/` 目录主要是数据的加载器和迭代器.

## 用法

克隆仓库代码后进入项目目录:
```bash
git clone https://github.com/FareedKhan-dev/train-llm-from-scratch.git
cd train-llm-from-scratch
```

如果你在导入依赖的时候遇到报错, 先确保pythonpath指向项目的根目录:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/train-llm-from-scratch"

# or if you are already in the train-llm-from-scratch directory
export PYTHONPATH="$PYTHONPATH:."
```

安装依赖:
```bash
pip install -r requirements.txt
```

你可以在 `src/models/transformer.py` 文件中修改编解码架构 以及 在 `config/config.py` 文件中修改训练配置.


下载训练数据集, 运行以下代码:
```bash
python scripts/data_download.py
```

脚本提供以下参数:
* `--train_max`: 指定下载训练文件的最大数量. 默认为 1 (最大为 30) 每个文件大约 11 GB.
* `--train_dir`: 指定训练数据的保存目录. 默认为 `data/train`.
* `--val_dir`: 指定校验数据的保存目录. 默认为 `data/val`.

对下载数据进行预训练,运行以下代码:
```bash
python scripts/data_preprocess.py
```

脚本提供以下参数:
- `--train_dir`: 指定训练数据保存目录 (默认为 `data/train`).
- `--val_dir`: 指定校验数据的保存目录 (默认为 `data/val`).
- `--out_train_file`: 指定预训练 HDF5 格式的训练数据的存储目录 (默认为 `data/train/pile_train.h5`).
- `--out_val_file`: 指定预训练后 HDF5 格式的校验数据存储目录(默认为 `data/val/pile_dev.h5`).
- `--tokenizer_name`: 指定训练数据的分词器 (默认为 `r50k_base`).
- `--max_data`: 指定处理每个JSON数据集的最大([lines](#训练数据))数(训练和校验数据集). 默认为 1000.

现在数据已经预处理好了, 修改 `config/config.py` 中的默认配置后你就可以开始训练你的 13 百万 参数 LLM了:

```python
# 定义词表(token 词典总数量，LLM 常见 32000、50257)大小 以及 transformer 配置 (3 Billion)
VOCAB_SIZE = 50304          # 词表中不重复(唯一标记)的token总数
CONTEXT_LENGTH = 128        # 模型最大序列长度(上下文长度)
N_EMBED = 128               # 词嵌入空间维度
N_HEAD = 8                  # 每个编解码器中注意力机制数量
N_BLOCKS = 1               # 模型中编解码器的数量
```

开始训练模型, 运行以下代码:
```bash
python scripts/train_transformer.py
```

运行后程序会开始训练模型, 并将模型保存到 `models/` 默认目录或者你在配置文件中修改后的目录.

要使用训练好的模型生成文本, 运行以下代码:
```bash
python scripts/generate_text.py --model_path models/your_model.pth --input_text hi
```

脚本提供以下参数:
- `--model_path`: 指定训练好的模型路径.
- `--input_text`: 指定用于生成文本的初始提示词.
- `--max_new_tokens`: 生成token的最大数量 (默认为 100).

代码会使用指定的模型并根据提示词生成文本.

## 手把手代码解读

这部分为了想详细了解代码的人写的. 我会从导入依赖开始到训练模型和生成文本, 手把手教你们.

之前, 我在Medium发表过一篇关于使用Tiny Shakespeare数据集创建一个[2.3+ million-parameter](https://levelup.gitconnected.com/building-a-million-parameter-llm-from-scratch-using-python-f612398f06c2) LLM, 但是输出并不理想. 下面是输出的示例:

```bash
# 230万参数的LLM输出
ZELBETH:
Sey solmenter! tis tonguerered if
Vurint as steolated have loven OID the queend refore
Are been, good plmp:

Proforne, wiftes swleen, was no blunderesd a a quain beath!
Tybell is my gateer stalk smend as be matious dazest
```

我产生了一个想法, 如果我将transformer架构简化和小型化, 同时将训练数据多样化,会怎么样?那么接下来, 个人开发者使用他们几乎坏掉的GPU训练出一个多大(参数)的模型, 才能支持生成正确的语法和语义呢?

我发现 **13+ million-parameter** 模型就足够生成正确的语法和标点, 这一发现令人振奋. 这意味着我们可以使用一个更具体的数据进一步微调我们之前训练好的模型. 最终我们可能得到一个参数在 1 billion以下或者甚至 500 million左右, 并且在特殊用途上表现的非常好的模型, 特别针对想要安全的私密数据(不需要再将数据喂给大模型,而是训练一个模型).

如果你本地GPU性能不够强到去训练十亿-参数的模型, 我建议你使用我仓库的代码 **先训练一个 1300+ 万参数** 的模型. 这样子你在一天内就能得到一个模型, 而不是花一段更长的时间. 

### 依赖导入

让我们先导入本次使用必备的库:

```python
# Pytorch中深度学习的方法和张量 
import torch
import torch.nn as nn
import torch.nn.functional as F

# 数据运算和数组处理
import numpy as np

# 处理HDF5文件
import h5py

# 操作系统和文件的包
import os

# 命令行参数解析
import argparse

# HTTP请求交互
import requests

# 循环进度条
from tqdm import tqdm

# JSON 数据处理
import json

# Zstandard 压缩库
import zstandard as zstd

# 大语言模型分词库
import tiktoken

# 数学运算 (用于高级数学函数)
import math
```

### 准备训练数据

我们需要多样化的数据, 包含不同领域的信息, 而 Pile 刚好合适. 虽然它有 825 GB,但是我们只需要使用其中的一部分,大约5%~10%. 我们需要先将数据集下载下来,然后再考虑怎么使用.我们下载的版本可以在 [HuggingFace](https://huggingface.co/datasets/monology/pile-uncopyrighted) 中获取.

```python
# 下载验证数据集
!wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst

# 下载第一部分的训练数据集
!wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/00.jsonl.zst

# 下载第二部分的训练数据集
!wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/01.jsonl.zst

# 下载第三部分的训练数据集
!wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/02.jsonl.zst
```

下载会消耗一些时间, 你也可以限制训练数据集只使用一个文件 `00.jsonl.zst`, 而不是使用三个. 它已经被分割到train/val/test三个文件中.下载完成后, 确保这些文件正确的放在各自的文件夹中.

```python
import os
import shutil
import glob

# 定义文件夹结构
train_dir = "data/train"
val_dir = "data/val"

# 如果文件夹不存在就创建
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 放置所有的训练文件 (列入, 00.jsonl.zst, 01.jsonl.zst, ...)
train_files = glob.glob("*.jsonl.zst")
for file in train_files:
    if file.startswith("val"):
        # 放置所有的校验文件
        dest = os.path.join(val_dir, file)
    else:
        # 放置训练文件
        dest = os.path.join(train_dir, file)
    shutil.move(file, dest)

Our dataset is in the .jsonl.zst format, which is a compressed file format commonly used for storing large datasets. It combines JSON Lines (.jsonl), where each line represents a valid JSON object, with Zstandard (.zst) compression. Let's read a sample of one of the downloaded files and see how it looks.

in_file = "data/val/val.jsonl.zst"  # 校验文件夹路径

with zstd.open(in_file, 'r') as in_f:
    for i, line in tqdm(enumerate(in_f)):  # 读取最先的5行
        data = json.loads(line)
        print(f"Line {i}: {data}")  # 打印出原始数据查看
        if i == 2:
            break
```

上面的代码输出如下:

```python
#### OUTPUT ####
Line: 0 
{
  "text": "Effect of sleep quality ... epilepsy.",
  "meta": {
    "pile_set_name": "PubMed Abstracts"
  }
}

Line: 1
{
  "text": "LLMops a new GitHub Repository ...",
  "meta": {
    "pile_set_name": "Github"
  }
}
```

现在我们对数据集进行编码(分词处理). 我们的目标是训练出一个至少能够输出正确语法的LLM. 因此, 我们需要使用一个成熟的分词器. 我们决定使用 OpenAi 的开源分词器 `tiktoken`. 我们使用 ChatGPT (GPT-3) 模型的分词器 `r50k_base` 来对我们的数据集进行处理.

当我们对训练数据集和验证数据集进行处理的时候, 还需要创建一个方法避免分词重复. 

```python
def process_files(input_dir, output_file):
    """
    将指定文件夹的所有 .zst 文件进行处理, 然后保存分词后的 HDF5 文件
    
    参数:
        输入文件夹路径 (str): 包含所有 .zst 文件的文件夹.
        输出文件夹路径 (str): 保存所有输出的 HDF5 文件的文件夹.
    """
    with h5py.File(output_file, 'w') as out_f:
        # 在HDF5文件中创建 'tokens' 可扩展数据集
        dataset = out_f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i')
        start_index = 0

        # 迭代输入文件夹中所有的文件
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".jsonl.zst"):
                in_file = os.path.join(input_dir, filename)
                print(f"Processing: {in_file}")

                # 打开 .zst 文件并读取
                with zstd.open(in_file, 'r') as in_f:
                    # 遍历压缩文件中的所有数据
                    for line in tqdm(in_f, desc=f"Processing {filename}"):
                        # 将读入的数据转为JSON
                        data = json.loads(line)
                        
                        # 在文本末尾添加结束标记(end-of-text token)并编码
                        text = data['text'] + "<|endoftext|>"
                        encoded = enc.encode(text, allowed_special={'<|endoftext|>'})
                        encoded_len = len(encoded)

                        # 计算tokens的结束索引
                        end_index = start_index + encoded_len

                        # 扩大数据集然后保存编码后的tokens
                        dataset.resize(dataset.shape[0] + encoded_len, axis=0)
                        dataset[start_index:end_index] = encoded
                        
                        # 为下一个循环更新开始索引
                        start_index = end_index
```
关于这个方法, 有两个重要点需要说明:

 1. 我们将编码后的数据存储在 HDF5 文件中, 以便之后训练的时候可以快速访问. 

 2. 为每个文本添加 `<|endoftext|>` 结束标记, 向模型说明上下文已经结束. 这对于模型生成有意义的输出有很大的帮助.

接下来我们可以使用以下代码来对我们的训练和验证数据集进行编码:

```python
# 定义编码后的数据输出文件
out_train_file = "data/train/pile_train.h5"
out_val_file = "data/val/pile_dev.h5"

# 加载(GPT-3/GPT-2 Model)的分词器
enc = tiktoken.get_encoding('r50k_base')

# 处理训练数据
process_files(train_dir, out_train_file)

# 处理验证数据
process_files(val_dir, out_val_file)
```

现在让我们看一下我们编码后的数据:

```python
 with h5py.File(out_val_file, 'r') as file:
     # 访问 'tokens' 数据集
     tokens_dataset = file['tokens']
     
     # 输出数据集的 Dtype
     print(f"Dtype of 'tokens' dataset: {tokens_dataset.dtype}")
     
     # 加载并打印少量开头的数据集
     print("First few elements of the 'tokens' dataset:")
     print(tokens_dataset[:10])  # 输出开头的10个token
```

上面代码输出如下:

```python
#### OUTPUT ####
Dtype of 'tokens' dataset: int32

First few elements of the 'tokens' dataset:
[ 2725  6557    83 23105   157   119   229    77  5846  2429]
```
我们已经准备好了训练的数据集. 接下来我们要编写 transformer 的架构并深入理解其原理.

### 模型预览

让我们快速预览一下 transformer 架构是如何处理和理解文本的. 它的原理是
It works by breaking text into smaller pieces called tokens and predicting the next token in the sequence. A transformer has many layers, called transformer blocks, stacked on top of each other, with a final layer at the end to make the prediction.

Each transformer block has two main components:

* **Self-Attention Heads**: These figure out which parts of the input are most important for the model to focus on. For example, when processing a sentence, the attention heads can highlight relationships between words, such as how a pronoun relates to the noun it refers to.

* **MLP (Multi-Layer Perceptron)**: This is a simple feed-forward neural network. It takes the information emphasized by the attention heads and processes it further. The MLP has an input layer that receives data from the attention heads, a hidden layer that adds complexity to the processing, and an output layer that passes the results to the next transformer block.

Together, the attention heads act as the “what to think about” part, while the MLP is the “how to think about it” part. Stacking many transformer blocks allows the model to understand complex patterns and relationships in the text, but this is not always guaranteed.

Instead of looking at the original paper diagram, let’s visualize a simpler and easier architecture diagram that we will be coding.

![Transformer Architecture by [Fareed Khan](undefined)](https://cdn-images-1.medium.com/max/11808/1*QXmeA-H52C-p82AwawslbQ.png)

Let’s read through the flow of our architecture that we will be coding:

 1. Input tokens are converted to embeddings and combined with position information.

 2. The model has 64 identical transformer blocks that process data sequentially.

 3. Each block first runs multi-head attention to look at relationships between tokens.

 4. Each block then processes data through an MLP that expands and then compresses the data.

 5. Each step uses residual connections (shortcuts) to help information flow.

 6. Layer normalization is used throughout to stabilize training.

 7. The attention mechanism calculates which tokens should pay attention to each other.

 8. The MLP expands the data to 4x size, applies ReLU, and then compresses it back down.

 9. The model uses 16 attention heads to capture different types of relationships.

 10. The final layer converts the processed data into vocabulary-sized predictions.

 11. The model generates text by repeatedly predicting the next most likely token.

### 多层感知器 (MLP)

MLP is a fundamental building block within the transformer’s feed-forward network. Its role is to introduce non-linearity and learn complex relationships within the embedded representations. When defining an MLP module, an important parameter is n_embed, which defines the dimensionality of the input embedding.

The MLP typically consists of a hidden linear layer that expands the input dimension by a factor (often 4, which we will use), followed by a non-linear activation function, commonly ReLU. This structure allows our network to learn more complex features. Finally, a projection linear layer maps the expanded representation back to the original embedding dimension. This sequence of transformations enables the MLP to refine the representations learned by the attention mechanism.

![MLP by [Fareed Khan](undefined)](https://cdn-images-1.medium.com/max/4866/1*GXxiLMW4kUXqOEimBA7g0A.png)

```python
# --- MLP (Multi-Layer Perceptron) Class ---

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron with one hidden layer.

    This module is used within the Transformer block for feed-forward processing.
    It expands the input embedding size, applies a ReLU activation, and then projects it back
    to the original embedding size.
    """
    def __init__(self, n_embed):
        super().__init__()
        self.hidden = nn.Linear(n_embed, 4 * n_embed)  # Linear layer to expand embedding size
        self.relu = nn.ReLU()                        # ReLU activation function
        self.proj = nn.Linear(4 * n_embed, n_embed)  # Linear layer to project back to original size

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is batch size,
                              T is sequence length, and C is embedding size.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        x = self.forward_embedding(x)
        x = self.project_embedding(x)
        return x

    def forward_embedding(self, x):
        """
        Applies the hidden linear layer followed by ReLU activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the hidden layer and ReLU.
        """
        x = self.relu(self.hidden(x))
        return x

    def project_embedding(self, x):
        """
        Applies the projection linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the projection layer.
        """
        x = self.proj(x)
        return x
```

We just coded our MLP part, where the __init__ method initializes a hidden linear layer that expands the input embedding size (n_embed) and a projection layer that reduces it back. ReLU activation is applied after the hidden layer. The forward method defines the data flow through these layers, applying the hidden layer and ReLU via forward_embedding, and the projection layer via project_embedding.

### 单头注意力机制

The attention head is the core part of our model. Its purpose is to focus on relevant parts of the input sequence. When defining a Head module, some important parameters are head_size, n_embed, and context_length. The head_size parameter determines the dimensionality of the key, query, and value projections, influencing the representational capacity of the attention mechanism.

The input embedding dimension n_embed defines the size of the input to these projection layers. context_length is used to create a causal mask, ensuring that the model only attends to preceding tokens.

Within the Head, linear layers (nn.Linear) for key, query, and value are initialized without bias. A lower triangular matrix (tril) of size context_length x context_length is registered as a buffer to implement causal masking, preventing the attention mechanism from attending to future tokens.

![Single Head Attention by [Fareed Khan](undefined)](https://cdn-images-1.medium.com/max/5470/1*teNwEhicq9ebVURiMS8WkA.png)

```python
# --- Attention Head Class ---

class Head(nn.Module):
    """
    A single attention head.

    This module calculates attention scores and applies them to the values.
    It includes key, query, and value projections, and uses causal masking
    to prevent attending to future tokens.
    """
    def __init__(self, head_size, n_embed, context_length):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)   # Key projection
        self.query = nn.Linear(n_embed, head_size, bias=False) # Query projection
        self.value = nn.Linear(n_embed, head_size, bias=False) # Value projection
        # Lower triangular matrix for causal masking
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        """
        Forward pass through the attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        scale_factor = 1 / math.sqrt(C)
        # Calculate attention weights: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        # Apply causal masking
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        v = self.value(x)   # (B, T, head_size)
        # Apply attention weights to values
        out = attn_weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
```

Our attention head class’s __init__ method initializes linear layers for key, query, and value projections, each projecting the input embedding (n_embed) to head_size. A lower triangular matrix based on context_length is used for causal masking. The forward method calculates attention weights by scaling the dot product of the query and key, applies the causal mask, normalizes the weights using softmax, and computes the weighted sum of the values to produce the attention output.

### 多头注意力机制

To capture diverse relationships within the input sequence, we are going to use the concept of multi-head attention. The MultiHeadAttention module manages multiple independent attention heads operating in parallel.

The key parameter here is n_head, which determines the number of parallel attention heads. The input embedding dimension (n_embed) and context_length are also necessary to instantiate the individual attention heads. Each head processes the input independently, projecting it into a lower-dimensional subspace of size n_embed // n_head. By having multiple heads, the model can attend to different aspects of the input simultaneously.

![Multi Head Attention by [Fareed Khan](undefined)](https://cdn-images-1.medium.com/max/6864/1*fa-YjrZdtbpuCLp7An99dg.png)

```python
# --- Multi-Head Attention Class ---

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module combines multiple attention heads in parallel. The outputs of each head
    are concatenated to form the final output.
    """
    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])

    def forward(self, x):
        """
        Forward pass through the multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after concatenating the outputs of all heads.
        """
        # Concatenate the output of each head along the last dimension (C)
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return x
```

Now that we have defined the MultiHeadAttention class, which combines multiple attention heads, the __init__ method initializes a list of Head instances (a total of n_head), each with a head_size of n_embed // n_head. The forward method applies each attention head to the input x and concatenates their outputs along the last dimension, merging the information learned by each head.

### 编解码器

To create a billion-parameter model, we definitely need a deep architecture. For that, we need to code a transformer block and stack them. The key parameters of a block are n_head, n_embed, and context_length. Each block comprises a multi-head attention layer and a feed-forward network (MLP), with layer normalization applied before each and residual connections after each.

Layer normalization, parameterized by the embedding dimension n_embed, helps stabilize training. The multi-head attention mechanism, as described before, takes n_head, n_embed, and context_length. The MLP also utilizes the embedding dimension n_embed. These components work together to process the input and learn complex patterns.

![Transformer Block by [Fareed Khan](undefined)](https://cdn-images-1.medium.com/max/6942/1*uLWGajZc6StnQHfZjcb6eA.png)

```python
# --- Transformer Block Class ---

class Block(nn.Module):
    """
    A single Transformer block.

    This block consists of a multi-head attention layer followed by an MLP,
    with layer normalization and residual connections.
    """
    def __init__(self, n_head, n_embed, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the block.
        """
        # Apply multi-head attention with residual connection
        x = x + self.attn(self.ln1(x))
        # Apply MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_embedding(self, x):
        """
        Forward pass focusing on the embedding and attention parts.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output after MLP embedding and the residual.
        """
        res = x + self.attn(self.ln1(x))
        x = self.mlp.forward_embedding(self.ln2(res))
        return x, res
```

Our Block class represents a single transformer block. The __init__ method initializes layer normalization layers (ln1, ln2), a MultiHeadAttention module, and an MLP module, all parameterized by n_head, n_embed, and context_length.

The forward method implements the block's forward pass, applying layer normalization and multi-head attention with a residual connection, followed by another layer normalization and the MLP, again with a residual connection. The forward_embedding method provides an alternative forward pass focused on the attention and initial MLP embedding stages.

### 最终模型

So far, we have coded small components of the transformer model. Next, we integrate token and position embeddings with a series of transformer blocks to perform sequence-to-sequence tasks. To do that, we need to code several key parameters: n_head, n_embed, context_length, vocab_size, and N_BLOCKS.

vocab_size determines the size of the token embedding layer, mapping each token to a dense vector of size n_embed. The context_length parameter is important for the position embedding layer, which encodes the position of each token in the input sequence, also with dimension n_embed. The number of attention heads (n_head) and the number of blocks (N_BLOCKS) dictate the depth and complexity of the network.

These parameters collectively define the architecture and capacity of the transformer model, so let’s code it.

![Transformer Class by [Fareed Khan](undefined)](https://cdn-images-1.medium.com/max/5418/1*0XXd_R2EOhkKCQDfqUQg0w.png)

```python
# --- Transformer Model Class ---

class Transformer(nn.Module):
    """
    The main Transformer model.

    This class combines token and position embeddings with a sequence of Transformer blocks
    and a final linear layer for language modeling.
    """
    def __init__(self, n_head, n_embed, context_length, vocab_size, N_BLOCKS):
        super().__init__()
        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.attn_blocks = nn.ModuleList([Block(n_head, n_embed, context_length) for _ in range(N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos_idxs', torch.arange(context_length))

    def _pre_attn_pass(self, idx):
        """
        Combines token and position embeddings.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Sum of token and position embeddings.
        """
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward(self, idx, targets=None):
        """
        Forward pass through the Transformer.

        Args:
            idx (torch.Tensor): Input token indices.
            targets (torch.Tensor, optional): Target token indices for loss calculation. Defaults to None.

        Returns:
            tuple: Logits and loss (if targets are provided).
        """
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.view(B * T, C)
            targets = targets.view(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
        return logits, loss

    def forward_embedding(self, idx):
        """
        Forward pass focusing on the embedding and attention blocks.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            tuple: Output after attention blocks and the residual.
        """
        x = self._pre_attn_pass(idx)
        residual = x
        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)
        return x, residual

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens given a starting sequence.

        Args:
            idx (torch.Tensor): Initial sequence of token indices.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: The extended sequence of tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

Our Transformer class `__init__` method initializes token and position embedding layers (token_embed, position_embed), a sequence of Block modules (attn_blocks), a final layer normalization layer (layer_norm), and a linear layer for language modeling (lm_head).

The _pre_attn_pass method combines token and position embeddings. The forward method processes the input sequence through the embedding layers and the series of transformer blocks, applies final layer normalization, and generates logits. It also calculates the loss if targets are provided. The forward_embedding method provides an intermediate forward pass up to the output of the attention blocks, and the generate method implements token generation.

### 训练批次

When we train a deep learning model on big data, we process it in batches due to GPU availability. So, let’s create a get_batch_iterator function, taking the data_path to an HDF5 file, the desired batch_size, the context_length for each sequence, and the device to load the data onto.

The batch_size determines how many sequences are processed in parallel during training, while the context_length specifies the length of each input sequence. The data_path points to the location of the training data.

```python
# --- Data Loading Utility --- 

def get_batch_iterator(data_path, batch_size, context_length, device="gpu"):
    """
    Creates an iterator for generating batches of data from an HDF5 file.

    Args:
        data_path (str): Path to the HDF5 file containing tokenized data.
        batch_size (int): Number of sequences in each batch.
        context_length (int): Length of each sequence.
        device (str, optional): Device to load the data onto ('cpu' or 'cuda'). Defaults to "cpu".

    Yields:
        tuple: A tuple containing input sequences (xb) and target sequences (yb).
    """
    # Open the HDF5 file in read mode
    with h5py.File(data_path, 'r') as hdf5_file:
        
        # Extract the dataset of tokenized sequences
        dataset = hdf5_file['tokens']
        
        # Get the total size of the dataset
        dataset_size = dataset.shape[0]
        
        # Calculate the number of examples (sequences) that can be made from the data
        n_examples = (dataset_size - 1) // context_length
        
        # Create an array of indices for examples and shuffle them for randomness
        example_idxs = np.arange(n_examples)
        np.random.shuffle(example_idxs)
        
        # Initialize epoch counter and example counter
        epochs = 0
        counter = 0
        
        while True:
            # Check if the current batch exceeds the number of available examples
            if counter + batch_size > n_examples:
                # Shuffle the indices again and reset the counter to 0
                np.random.shuffle(example_idxs)
                counter = 0
                print(f"Finished epoch {epochs}")  # Print epoch number when an epoch finishes
                epochs += 1  # Increment the epoch counter
            
            # Select a batch of random indices to generate sequences
            random_indices = example_idxs[counter:counter+batch_size] * context_length
            
            # Retrieve sequences from the dataset based on the random indices
            random_samples = torch.tensor(np.array([dataset[idx:idx+context_length+1] for idx in random_indices]))
            
            # Separate the input sequences (xb) and target sequences (yb)
            xb = random_samples[:, :context_length].to(device)  # Input sequence (first half of the random sample)
            yb = random_samples[:, 1:context_length+1].to(device)  # Target sequence (second half of the random sample)
            
            # Increment the counter to move to the next batch
            counter += batch_size
            
            # Yield the input and target sequences as a tuple for the current batch
            yield xb, yb
```
Our get_batch_iterator function handles the loading and batching of training data. It takes data_path, batch_size, context_length, and device as input. The function opens the HDF5 file, shuffles the data, and then enters an infinite loop to generate batches. In each iteration, it selects a random subset of the data to form a batch of input sequences (xb) and their corresponding target sequences (yb).

### 训练参数

Now that we have coded our model, we need to define the training parameters, such as the number of heads, blocks, and more, along with the data path.

```python
# --- Configuration ---

# Define vocabulary size and transformer configuration
VOCAB_SIZE = 50304          # Number of unique tokens in the vocabulary
CONTEXT_LENGTH = 512        # Maximum sequence length for the model
N_EMBED = 2048              # Dimension of the embedding space
N_HEAD = 16                 # Number of attention heads in each transformer block
N_BLOCKS = 64               # Number of transformer blocks in the model

# Paths to training and development datasets
TRAIN_PATH = "data/train/pile_val.h5"  # File path for the training dataset
DEV_PATH = "data/val/pile_val.h5"      # File path for the validation dataset

# Transformer training parameters
T_BATCH_SIZE = 32          # Number of samples per training batch
T_CONTEXT_LENGTH = 16      # Context length for training batches
T_TRAIN_STEPS = 200000     # Total number of training steps
T_EVAL_STEPS = 1000        # Frequency (in steps) to perform evaluation
T_EVAL_ITERS = 250         # Number of iterations to evaluate the model
T_LR_DECAY_STEP = 50000    # Step at which to decay the learning rate
T_LR = 5e-4                # Initial learning rate for training
T_LR_DECAYED = 5e-5        # Learning rate after decay
T_OUT_PATH = "models/transformer_B.pt"  # Path to save the trained model

# Device configuration
DEVICE = 'cuda'

# Store all configurations in a dictionary for easy access and modification
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
}
```

For most of the parameters, I have used the most common values and also stored them in a dictionary for easy access. Here, the parameters are for a billion-parameter model. If you want to train a model with millions of parameters, you can reduce the main parameters, which include CONTEXT_LENGTH, N_EMBED, N_HEAD, and N_BLOCKS. However, you can also run the million-parameter model script in my GitHub repository.

### 模型训练

Let's initialize our transformer model and check its total number of parameters.
```python
# --- Initialize the Model and Print Parameters --- 

model = Transformer(
    n_head=config['n_head'],
    n_embed=config['n_embed'],
    context_length=config['context_length'],
    vocab_size=config['vocab_size'],
    N_BLOCKS=config['n_blocks']
).to(config['device'])


# Print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")


#### OUTPUT ####
2,141,346,251
```

Now that we have 2 Billion parameter model, we need to define our Adam optimizer and loss tracking function, which will help us track the progress of our model throughout the training.

```python
# --- Optimizer Setup and Loss Tracking --- 

# Set up the AdamW optimizer with the specified learning rate.
optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])

# List to track loss values during training.
losses = []

# Define a window size for averaging recent losses in the training loop.
AVG_WINDOW = 64

# Helper function to estimate the average loss for training and development data.
@torch.no_grad()
def estimate_loss(steps):
    """
    Evaluate the model on training and development datasets and calculate average loss.

    Args:
        steps (int): Number of steps to evaluate.

    Returns:
        dict: Dictionary containing average losses for 'train' and 'dev' splits.
    """
    out = {}
    model.eval()  # Set the model to evaluation mode.

    for split in ['train', 'dev']:
        # Select the appropriate data path for the current split.
        data_path = config['train_path'] if split == 'train' else config['dev_path']
        
        # Create a batch iterator for evaluation.
        batch_iterator_eval = get_batch_iterator(
            data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
        )
        
        # Initialize a tensor to track loss values for each evaluation step.
        losses_eval = torch.zeros(steps)
        for k in range(steps):
            try:
                # Fetch a batch and calculate the loss.
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval[k] = loss.item()
            except StopIteration:
                # Handle the case where the data iterator ends early.
                print(f"Warning: Iterator for {split} ended early.")
                break
        
        # Compute the mean loss for the current split.
        out[split] = losses_eval[:k + 1].mean()
    
    model.train()  # Restore the model to training mode.
    return out
```

We will now initialize our batch processing function and training loop, which will start our training.

```python
# --- Training Loop ---

# Create a batch iterator for the training data.
batch_iterator = get_batch_iterator(
  config['train_path'],
  config['t_batch_size'],
  config['t_context_length'],
  device=config['device']
)

# Create a progress bar to monitor training progress.
pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
  try:
      # Fetch a batch of input and target data.
      xb, yb = next(batch_iterator)
      
      # Perform a forward pass and compute the loss.
      _, loss = model(xb, yb)
      
      # Record the loss for tracking.
      losses.append(loss.item())
      pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")
      
      # Backpropagate the loss and update the model parameters.
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      # Periodically evaluate the model on training and development data.
      if step % config['t_eval_steps'] == 0:
          train_loss, dev_loss = estimate_loss(config['t_eval_iters']).values()
          print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")

      # Decay the learning rate at the specified step.
      if step == config['t_lr_decay_step']:
          print('Decaying learning rate')
          for g in optimizer.param_groups:
              g['lr'] = config['t_lr_decayed']
  except StopIteration:
      # Handle the case where the training data iterator ends early.
      print("Training data iterator finished early.")
      break
```
### 保存训练模型

Since our training loop has the ability to handle errors, in case the loop throws any error, it will save our partially trained model to avoid loss. Once the training is complete, we can save our trained model to use it later for inference.

```python
# --- Save Model and Final Evaluation ---

# Perform a final evaluation of the model on training and development datasets.
train_loss, dev_loss = estimate_loss(200).values()

# Ensure unique model save path in case the file already exists.
modified_model_out_path = config['t_out_path']
save_tries = 0
while os.path.exists(modified_model_out_path):
    save_tries += 1
    model_out_name = os.path.splitext(config['t_out_path'])[0]
    modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"

# Save the model's state dictionary, optimizer state, and training metadata.
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'steps': len(losses),
    },
    modified_model_out_path
)
print(f"Saved model to {modified_model_out_path}")
print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
```
The final training loss for the billion-parameter model is 0.2314, and the dev loss is 0.643.

### 训练损失

When I plot the loss of both the million- and billion-parameter models, they look very different.

![Training Loss Comparison](https://cdn-images-1.medium.com/max/6696/1*8Gl7cEbainB4GRVwL3cc7Q.png)

The billion-parameter model starts with a much higher loss and fluctuates a lot at the beginning. It goes down quickly at first, but then wobbles before becoming smoother. This shows that the bigger model has a harder time finding the right way to learn at the start. It might need more data and careful settings. When the learning rate is lowered (the red line), the loss goes down more steadily, showing that this helps it fine-tune.

The million-parameter model’s loss goes down more easily from the start. It doesn’t fluctuate as much as the bigger model. When the learning rate is lowered, it doesn’t change the curve as much. This is likely because the smaller model is simpler to train and finds a good solution faster. The big difference shows how much harder it is to train very large models. They need different methods and maybe more time to learn well.

We now have our saved model. We can finally use it for inference and see how it generates text. 😓

### 文本生成

Let’s create a function to generate text from our saved model, which takes the saved model path and the encoder as inputs and returns the generated text.

```python
def generate_text(model_path, input_text, max_length=512, device="gpu"):
    """
    Generate text using a pre-trained model based on the given input text.

    Args:
    - model_path (str): Path to the model checkpoint.
    - device (torch.device): Device to load the model on (e.g., 'cpu' or 'cuda').
    - input_text (str): The input text to seed the generation.
    - max_length (int, optional): Maximum length of generated text. Defaults to 512.

    Returns:
    - str: The generated text.
    """

    # Load the model checkpoint
    checkpoint = torch.load(model_path)

    # Initialize the model (you should ensure that the Transformer class is defined elsewhere)
    model = Transformer().to(device)

    # Load the model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load the tokenizer for the GPT model (we use 'r50k_base' for GPT models)
    enc = tiktoken.get_encoding('r50k_base')

    # Encode the input text along with the end-of-text token
    input_ids = torch.tensor(
        enc.encode(input_text, allowed_special={'<|endoftext|>'}),
        dtype=torch.long
    )[None, :].to(device)  # Add batch dimension and move to the specified device

    # Generate text with the model using the encoded input
    with torch.no_grad():
        # Generate up to 'max_length' tokens of text
        generated_output = model.generate(input_ids, max_length)

        # Decode the generated tokens back into text
        generated_text = enc.decode(generated_output[0].tolist())

    return generated_text
```

The transformer we defined earlier needs to be called here to load the architecture, and then we load the saved model as the state in that architecture.

Let’s first observe what both the million and billion-parameter models generate without providing any input, and see what they generate randomly.

```python
# Defining the file paths for the pre-trained models
Billion_model_path = 'models/transformer_B.pt'  # Path to the Billion model
Million_model_path = 'models/transformer_M.pt'  # Path to the Million model

# Using '<|endoftext|>' as input to the models (acts as a prompt that allows the models to generate text freely)
input_text = "<|endoftext|>"

# Call the function to generate text based on the input text using the Billion model
B_output = generate_text(Billion_model_path, input_text)

# Call the function to generate text based on the input text using the Million model
M_output = generate_text(Million_model_path, input_text)

# Print the output generated by both models
print(B_output)  # Output from the Billion model
print(M_output)  # Output from the Million model
```

| **Million Parameter Output** | **Billion Parameter Output** |
|------------------------------|------------------------------|
| In 1978, The park was returned to the factory-plate that the public share to the lower of the electronic fence that follow from the Station's cities. The Canal of ancient Western nations were confined to the city spot. The villages were directly linked to cities in China that revolt that the US budget and in Odambinais is uncertain and fortune established in rural areas. | There are two miles east coast from 1037 and 73 million refugees (hypotetus) as the same men and defeated Harvard, and Croft. At right east and West Nile's Mediterranean Sea jets. It was found there a number of parties, blacksmith, musician and boutique hospitality and inspire the strain delivered Canadians have already killed, rural branches with coalition railholder against Abyssy. |


Both LLMs are able to generate clear and accurate words when the context is short and simple. For example, in the million-parameter output, the phrase **“The villages were directly linked to cities in China”** makes sense and conveys a clear idea. It is easy to understand and logically connects the villages to the cities.

However, when the context becomes longer and more complex, the clarity begins to fade. In the billion-parameter output, sentences like **“There are two miles east coast from 1037 and 73 million refugees (hypotetus)”** and **“blacksmith, musician and boutique hospitality and inspire the strain delivered Canadians”** become harder to follow. The ideas seem disjointed, and the sentence structure doesn’t flow naturally. While the words used might still be correct, the overall meaning becomes confusing and unclear.

The positive point is that the 13+ million-parameter LLM also starts generating some kind of meaningful content with correct word spelling. For instance, when I use the subject input text, it starts generating an email for me. Although, obviously, broader text doesn’t provide meaningful results, take a look at the output:

```python
# Input text
input_text "Subject: "

# Call the Million parameter Mod
m_output = generate_text(Million_model_path, input_text)

print(m_output)  # Output from the Million model
```
| **Million Parameter LLM Output**                                                                 |
|--------------------------------------------------------------------------------------------------|
| Subject: ClickPaper-summary Study for Interview <br>Good morning, I hope this message finds you well, as the sun gently peeks through the clouds, ... |

Our million parameter model gives us the motivation that we can have a very narrow, goal-oriented LLM under 1B in size, while our 1B trained model shows us that the architecture needs to be coded in great depth with proper consideration. Otherwise, it won’t improve training or performance compared to the million-parameter model. It will just overfit the data unless you have a deep architecture for the billion-sized model.

# 接下来呢

I recommend that you create the 13+ million-parameter model and then start scaling it by adding the next 100 parameters, improving its ability to handle shorter contexts. It’s up to you how many more parameters you want to train for specific tasks. Then, for the remaining parameters under 1B, try fine-tuning the model on domain-specific data, such as writing emails or essays, and see how it generates the text.

<hr>

Wanna chat on something? [My Linkedin](https://www.linkedin.com/in/fareed-khan-dev/)

## Star History

[![](https://api.star-history.com/svg?repos=FareedKhan-dev/train-llm-from-scratch&type=Date)](https://star-history.com/#FareedKhan-dev/train-llm-from-scratch&Date)
