# DeepBait — Article-Conditioned Clickbait Headline Generator

ECE1508 Applied Deep Learning — Group 23
Irys Zhang, Jiangchuan Yu, Yushun Tang

A deep learning system that reads a news article and generates a clickbait-style headline for it. We compare three training strategies: direct LSTM training, two-stage LSTM pretraining + fine-tuning, and BART transformer fine-tuning.

---

## Project Structure

```
deepbait/
├── src/
│   ├── data_processing.py    # Data loading, tokenization, vocabulary
│   ├── model.py              # Seq2SeqClickbait (ArticleEncoder + ClickbaitDecoder)
│   ├── train.py              # LSTM training loop
│   ├── generate.py           # LSTM headline generation with temperature sampling
│   └── evaluate.py           # Perplexity evaluation + headline ranking
├── scripts/
│   ├── build_vocab.py        # Build shared vocabulary across all data sources
│   ├── run_direct.py         # Experiment 1: direct clickbait training
│   ├── run_pretrain_finetune.py  # Experiment 2: two-stage training
│   ├── download_bart.py      # Download BART model from HuggingFace
│   ├── run_bart_finetune.py  # Experiment 3: BART fine-tuning
│   └── generate_bart.py      # BART headline generation
├── notebooks/
│   └── demo.ipynb            # End-to-end demo notebook
├── docs/
│   └── report.tex            # LaTeX project report
├── checkpoints/              # Saved model checkpoints (gitignored)
├── data/                     # Datasets (gitignored)
├── models/                   # Downloaded BART weights (gitignored)
├── presentation/             # Presentation slides
├── pyproject.toml            # Project config (uv)
├── requirements.txt          # Pip dependencies
└── README.md
```

---

## Setup

### Option A — uv (recommended)

[uv](https://docs.astral.sh/uv/) handles Python version management and virtual environments automatically.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv automatically)
uv sync
```

All commands below use `uv run` to run inside the managed environment.

### Option B — pip

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install numpy pandas matplotlib jupyter ipykernel
```

### GPU Support

The `pyproject.toml` is configured for CUDA 12.6. If you have a different CUDA version or want CPU-only, adjust the `[tool.uv.sources]` section or install PyTorch manually following [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Data Preparation

Two datasets are used:

| Source | Size | Description |
|--------|------|-------------|
| [DL in NLP Spring 2019 Classification](https://kaggle.com/competitions/dlinnlp-spring-2019-clf) | ~24K total, ~3.7K clickbait | Kaggle competition dataset |
| [Webis-Clickbait-17](https://webis.de/data/webis-clickbait-17.html) | ~4.7K clickbait | Validation set, `truthMean >= 0.5` |

### Step 1: Kaggle Dataset

Download from Kaggle and place in `data/`:

```bash
mkdir -p data
# Option A: Kaggle API
pip install kaggle
kaggle competitions download -c dlinnlp-spring-2019-clf -p data/
unzip data/dlinnlp-spring-2019-clf.zip -d data/

# Option B: Manual
# Download from https://kaggle.com/competitions/dlinnlp-spring-2019-clf
# Place train.csv into data/
```

The CSV must contain columns: `title` (headline), `text` (article body), `label` (1 = clickbait).

### Step 2: Webis-Clickbait-17 (optional but recommended)

Download the **validation set JSON files** from [Zenodo](https://zenodo.org/records/5530410) (do NOT download the 96 GB WARC archives):

```bash
mkdir -p data/webis17
# Extract so that data/webis17/clickbait17-validation-170630/ contains:
#   instances.jsonl
#   truth.jsonl
```

### Step 3: Build Shared Vocabulary (for LSTM experiments)

All LSTM experiments should share the same vocabulary for fair comparison:

```bash
uv run python scripts/build_vocab.py

# Quick test with fewer HuggingFace samples:
uv run python scripts/build_vocab.py --hf_max_samples 50000
```

This builds vocabulary from CNN/DailyMail + Kaggle + Webis data and saves it to `checkpoints/shared_vocab.pt`.

---

## Running Experiments

### Experiment 1: Direct LSTM Training

Train the LSTM encoder-decoder from scratch on clickbait pairs only.

```bash
uv run python scripts/run_direct.py

# With a different random seed:
uv run python scripts/run_direct.py --split_seed 17

# Custom settings:
uv run python scripts/run_direct.py --epochs 100 --patience 10 --lr 1e-3
```

Checkpoints are saved to `checkpoints/exp1_direct/`.

### Experiment 2: Two-Stage LSTM Training

Stage 1 pretrains on all articles (general language), Stage 2 fine-tunes on clickbait only.

```bash
uv run python scripts/run_pretrain_finetune.py

# With a different random seed:
uv run python scripts/run_pretrain_finetune.py --split_seed 17

# Quick test with fewer HuggingFace samples:
uv run python scripts/run_pretrain_finetune.py --hf_max_samples 30000

# Custom stage settings:
uv run python scripts/run_pretrain_finetune.py \
    --pretrain_epochs 10 \
    --finetune_epochs 30 \
    --finetune_lr 1e-4
```

Checkpoints are saved to `checkpoints/exp2_pretrain/` and `checkpoints/exp2_finetune/`.

### Experiment 3: BART Fine-Tuning

Fine-tune the pre-trained BART-Large-CNN transformer.

```bash
# Step 1: Download BART model (~1.6 GB)
uv run python scripts/download_bart.py

# Step 2: Fine-tune on clickbait data
uv run python scripts/run_bart_finetune.py

# With a different random seed:
uv run python scripts/run_bart_finetune.py --split_seed 17

# Fewer epochs for a quick test:
uv run python scripts/run_bart_finetune.py --epochs 2
```

Checkpoints are saved to `checkpoints/exp3_bart/`.

### Multi-Seed Experiments

To reproduce the full experimental setup from the report, run each experiment with seeds {42, 17, 99}:

```bash
for seed in 42 17 99; do
    uv run python scripts/run_direct.py \
        --split_seed $seed \
        --checkpoint_dir checkpoints/exp1_seed${seed}

    uv run python scripts/run_pretrain_finetune.py \
        --split_seed $seed \
        --pretrain_dir checkpoints/exp2_pretrain_seed${seed} \
        --finetune_dir checkpoints/exp2_finetune_seed${seed}

    uv run python scripts/run_bart_finetune.py \
        --split_seed $seed \
        --output_dir checkpoints/exp3_bart_seed${seed}
done
```

---

## Generating Headlines

### LSTM Models

```bash
uv run python src/generate.py \
    --checkpoint checkpoints/exp2_finetune/best_model.pt \
    --article "Scientists at MIT have developed a new AI system that can diagnose rare diseases from blood samples." \
    --temperature 0.8 \
    --num_headlines 5
```

### BART Model

```bash
# From a CLI argument:
uv run python scripts/generate_bart.py \
    --model_dir checkpoints/exp3_bart/best_model \
    --article "Scientists at MIT have developed a new AI system that can diagnose rare diseases from blood samples."

# From a text file:
uv run python scripts/generate_bart.py \
    --model_dir checkpoints/exp3_bart/best_model \
    --article_file test_article.txt

# Interactive mode:
uv run python scripts/generate_bart.py \
    --model_dir checkpoints/exp3_bart/best_model \
    --interactive
```

### Temperature Guide (LSTM)

| Value | Effect |
|-------|--------|
| 0.5 | Conservative, repetitive |
| 0.8 | Balanced (recommended) |
| 1.0 | Standard sampling |
| 1.2+ | Creative, less coherent |

---

## Low-Level Training API

The `src/train.py` script provides fine-grained control for custom experiments:

```bash
uv run python src/train.py \
    --data_path data/train.csv \
    --webis17_path data/webis17/clickbait17-validation-170630 \
    --checkpoint_dir checkpoints/custom_run \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-3 \
    --embed_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --max_article_len 100 \
    --max_title_len 20 \
    --split_seed 42

# Pre-training (all articles, no clickbait filter):
uv run python src/train.py \
    --data_path data/train.csv \
    --no_clickbait_filter \
    --hf_dataset cnn_dailymail:3.0.0 \
    --checkpoint_dir checkpoints/pretrain \
    --epochs 10

# Fine-tuning from a pre-trained checkpoint:
uv run python src/train.py \
    --data_path data/train.csv \
    --webis17_path data/webis17/clickbait17-validation-170630 \
    --resume_checkpoint checkpoints/pretrain/best_model.pt \
    --checkpoint_dir checkpoints/finetune \
    --epochs 30 \
    --lr 1e-4
```

---

## Model Architecture

### LSTM Encoder-Decoder (~8M parameters)

```
Article → Embedding(128) → LSTM(256, 2 layers) → (h_n, c_n)
                                                       ↓
Title   ← Linear(vocab) ← LSTM(256, 2 layers) ← Embedding(128)
```

- **Loss:** Cross-entropy (PAD tokens ignored)
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
- **Gradient clipping:** max norm = 1.0
- **Teacher forcing** during training

### BART-Large-CNN (~406M parameters)

Pre-trained transformer with 12 encoder + 12 decoder layers, fine-tuned with HuggingFace `Seq2SeqTrainer`. Uses BPE tokenization, FP16 mixed precision, and early stopping.

---

## Demo Notebook

```bash
uv run jupyter notebook notebooks/demo.ipynb
```

The notebook demonstrates loading a trained model, generating headlines for sample articles, and visualising training curves.

---

## Team Contributions

- **Irys Zhang:** Dataset preprocessing, tokenization pipeline, vocabulary construction, Webis-17 data integration (`data_processing.py`)
- **Jiangchuan Yu:** Neural network architecture, training loops, BART fine-tuning (`model.py`, `train.py`, `run_bart_finetune.py`)
- **Yushun Tang:** Text generation, temperature sampling, evaluation scripts, demo notebook (`generate.py`, `evaluate.py`, `demo.ipynb`)
- **All members:** Experimental design, hyperparameter tuning, cross-seed experiments, results analysis, report writing, presentation
