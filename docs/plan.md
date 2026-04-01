# DeepBait 改造方案

## 1. 问题背景

原始实现训练了一个 LSTM 语言模型，输入为空或 seed phrase，直接从零生成标题党标题。这个方案没有实际意义——生成的标题与任何真实文章内容无关。

**目标：改为输入一篇文章，输出一个针对该文章的标题党标题。**

---

## 2. 架构改造：Encoder-Decoder

### 原架构

```
<START> → LSTM Decoder → 标题词1 → 标题词2 → ...
（无任何文章输入）
```

### 新架构

```
文章正文
  ↓ tokenize + truncate（前 100 词）
  ↓ Embedding
  ↓ LSTM Encoder
  → 最终隐状态 (h_n, c_n)
        ↓ 初始化 Decoder
<START> → LSTM Decoder → 标题词1 → 标题词2 → ...
```

文章经 Encoder 压缩成上下文向量，注入 Decoder 的初始隐状态，Decoder 再自回归地生成标题。Encoder 和 Decoder 共享同一套词表、相同的 `embed_dim` 和 `hidden_dim`，使隐状态可直接传递。

### 训练方式：Teacher Forcing

训练时 Decoder 每一步接收**真实的上一个词**（而非自己预测的词），加速收敛：

```
Encoder 输入:  [文章 token 序列]
Decoder 输入:  [<START>, w1, w2, w3, <PAD>, ...]
Decoder 目标:  [w1,      w2, w3, <END>, <PAD>, ...]
```

Loss 对 `<PAD>` 位置忽略，只计算有效 token。

---

## 3. 代码改动

### `src/data_processing.py`

| 改动 | 说明 |
|------|------|
| `load_clickbait()` → `load_article_title_pairs()` | 同时读取 `text` 列（文章）和 `title` 列，原实现忽略了 `text` 列 |
| `build_vocab()` | 接受任意文本列表，从 articles + titles 合并建词表 |
| `ClickbaitDataset` → `ArticleTitleDataset` | 每个样本返回三个 tensor：`(article, dec_input, target)` |
| `build_dataloaders()` | 新增 `max_article_len` 参数，返回类型更新 |

### `src/model.py`

| 新增类 | 说明 |
|--------|------|
| `ArticleEncoder` | Embedding + LSTM，输出最终隐状态 `(h_n, c_n)` |
| `ClickbaitDecoder` | 原 `ClickbaitLSTM` 重命名，接受外部初始隐状态 |
| `Seq2SeqClickbait` | 组合 Encoder + Decoder；训练用 `forward()`，推理用 `encode()` + `decode_step()` |

### `src/train.py`

- `train_epoch` / `eval_epoch` 改为处理三元组 batch `(article, dec_input, target)`
- Loss 对整个 title 序列计算：`criterion(logits.reshape(-1, vocab_size), target.reshape(-1))`
- Hyperparams 增加 `max_article_len`

### `src/generate.py`

- `load_model()` 重建 `Seq2SeqClickbait`
- `generate_headline()` 必填 `article_text`，先 encode 文章，再自回归 decode
- CLI 参数 `--seed` → `--article`

---

## 4. 数据情况

### 现有数据

| 文件 | 总行数 | Clickbait 行数 | 可用于训练？ |
|------|--------|--------------|------------|
| `data/train.csv` | 24,871 | 3,748 | 已用 |
| `data/valid.csv` | 3,552 | 543 | 可合并 |
| `data/test.csv` | 5,647 | 无标签 | 不可用 |
| `data/unlabeled.csv` | 80,013 | 无标签 | 不建议直接用 |

合并 `train.csv` + `valid.csv` 可得 **4,291 条** clickbait `(article, title)` 对。

### 问题分析

首次训练结果（仅用 train.csv）：

```
Epoch  2: Train PPL 1431 → Val PPL 1540  ← 最优点
Epoch 20: Train PPL  593 → Val PPL 1596  ← 严重过拟合
```

根本原因：

1. **数据量严重不足**：3,352 条训练样本，训练 22M 参数的模型
2. **词表过大**：文章正文引入大量低频词，`min_freq=2` 导致词表 39K（原来 15K）
3. **两件事同时从零学**：模型既要学"读文章"，又要学"写标题党"，数据不够支撑

---

## 5. 推荐数据方案：引入 Webis Clickbait Corpus 2017

### 为什么选它

这是唯一同时满足以下条件的公开数据集：
- 有完整文章正文（`targetParagraphs` 字段，已预处理为 JSON）
- 有对应的 clickbait 标题（`postText` 字段）
- 有 clickbait 评分（0-1），可筛选高质量样本
- 英文，与现有数据集语言一致

### 数据量对比

| 数据来源 | 样本数 |
|---------|--------|
| 现有 clickbait 数据 | 4,291 |
| Webis Clickbait Corpus 2017 | ~38,500 |
| **合并后总计** | **~42,800** |

样本量提升约 **10 倍**，且全部为 clickbait 风格，无需两阶段训练。

### 下载与格式

- 来源：Zenodo，搜索 "Webis-Clickbait-17"
- 只需下载 JSON 文件（无需下载 96GB 的 WARC 原始爬取文件）
- 关键字段：

```json
{
  "postText": ["标题党标题"],
  "targetParagraphs": ["文章段落1", "文章段落2", ...],
  "truthMean": 0.8
}
```

### 数据加载适配

在 `data_processing.py` 中新增 `load_webis17()` 函数，将上述字段转换为 `(article_text, title)` 元组，与现有 `load_article_title_pairs()` 返回格式一致，然后合并两个列表即可。

---

## 6. 调参建议

在引入更多数据的同时，配合以下参数调整：

| 参数 | 当前值 | 建议值 | 原因 |
|------|--------|--------|------|
| `min_freq` | 2 | **10** | 压缩词表，减少噪声 |
| `embed_dim` | 128 | 128 | 保持 |
| `hidden_dim` | 256 | 256 | 保持 |
| `max_article_len` | 100 | 100 | 保持 |
| `dropout` | 0.3 | **0.4-0.5** | 数据更多后可适当增强正则 |
| `epochs` | 20 | 20-30 | 视 val loss 曲线决定 |

---

## 7. 训练环境建议

推荐在 RTX 4060 笔记本上运行，代码无需修改（已自动检测 CUDA）。

**预计训练时长（42K 样本 × 20 epoch）：**

| 设备 | 每 epoch | 20 epoch 总计 |
|------|---------|-------------|
| CPU（当前） | ~8 分钟 | ~2.5 小时 |
| RTX 4060 | ~20-30 秒 | **~10 分钟** |

Windows 上若 DataLoader 报错，将 `num_workers=2` 改为 `num_workers=0`。

---

## 8. 进度

- [x] `src/data_processing.py`：改为加载 `(article, title)` 对，新增 `ArticleTitleDataset`
- [x] `src/model.py`：新增 `ArticleEncoder`、`ClickbaitDecoder`、`Seq2SeqClickbait`
- [x] `src/train.py`：适配 Encoder-Decoder 训练流程（teacher forcing）
- [x] `src/generate.py`：改为输入文章文本，encode 后自回归生成标题
- [x] 在 CPU 上完成首次训练验证（20 epoch，确认流程跑通）
- [ ] 下载 Webis Clickbait Corpus 2017 JSON 文件
- [ ] 在 `data_processing.py` 中添加 `load_webis17()` 适配函数
- [ ] 合并两个数据源，重新建词表（`min_freq=10`）
- [ ] 在 RTX 4060 上重新训练
- [ ] 更新 `notebooks/demo.ipynb` 展示文章输入 → 标题党输出的完整流程
