# DeepBait — Automated Clickbait Headline Generator

ECE1508 Applied Deep Learning — Group 23
Irys Zhang · Jiangchuan Yu · Yushun Tang

An LSTM-based language model that learns the writing style of clickbait headlines and generates new ones from scratch.

---

## Project Structure

```
deepbait/
├── data/
│   ├── download_data.sh      # Kaggle download script
│   └── vocab.json            # Generated after first run
├── src/
│   ├── data_processing.py    # Data loading, tokenization, vocabulary
│   ├── model.py              # ClickbaitLSTM architecture
│   ├── train.py              # Training loop
│   ├── generate.py           # Headline generation with temperature sampling
│   └── evaluate.py           # Perplexity + qualitative evaluation
├── checkpoints/              # Saved model checkpoints (created at training)
├── outputs/                  # Generated headlines (created at evaluation)
├── notebooks/
│   └── demo.ipynb            # End-to-end demo
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

Download the **DL in NLP Spring 2019 Classification** dataset from Kaggle:

**Option A — Kaggle API (automated):**
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
bash data/download_data.sh
```

**Option B — Manual:**
1. Visit https://www.kaggle.com/datasets/datasnaek/clickbait
2. Download and extract the CSV into `data/`

The CSV must contain a text column (`headline` / `text`) and a label column (`label`).
Only rows with `label == 1` (clickbait) are used.

---

## Usage

All commands should be run from the project root.

### 1. Train

```bash
python src/train.py \
    --data_path data/train.csv \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --embed_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.3
```

Saves checkpoints to `checkpoints/` and a loss curve PNG.

### 2. Generate Headlines

```bash
# With a seed phrase
python src/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --seed "10 things" \
    --temperature 0.8 \
    --num_headlines 5

# Unconditional (no seed)
python src/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --temperature 1.0 \
    --num_headlines 10
```

**Temperature guide:**
| Value | Effect |
|-------|--------|
| 0.5   | Conservative, repetitive |
| 0.8   | Balanced (recommended) |
| 1.0   | Standard sampling |
| 1.2+  | Creative, less coherent |

### 3. Evaluate

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_path data/train.csv
```

Prints validation perplexity and saves 50 generated headlines (best & worst) to `outputs/generated_headlines.txt`.

### 4. Demo Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Model Architecture

```
Input tokens → Embedding (128-dim) → LSTM (256 hidden, 2 layers) → Linear → Softmax
```

- **Loss:** Cross-Entropy
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
- **Gradient clipping:** max norm = 1.0

---

## Team Contributions

- **Irys Zhang:** Dataset preprocessing, tokenization pipeline, vocabulary construction (`data_processing.py`)
- **Jiangchuan Yu:** Neural network architecture, training loop (`model.py`, `train.py`)
- **Yushun Tang:** Text generation, temperature sampling, evaluation scripts (`generate.py`, `evaluate.py`)
- **All members:** Experimentation, results analysis, report writing, presentation
