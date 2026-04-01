"""
data_processing.py
------------------
Dataset loading, text cleaning, tokenization, and vocabulary construction
for the Automated Clickbait Headline Generator (article -> title seq2seq).

Usage:
    python src/data_processing.py --data_path data/train.csv --output_dir data/
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2
END_IDX = 3


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation (keep apostrophes), collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split on whitespace."""
    return text.split()


def load_article_title_pairs(csv_path: str) -> list[tuple[str, str]]:
    """
    Load the dataset and return (article_text, title) pairs for clickbait rows.

    Expected CSV columns: a title column, an article body column, and a label column.
    """
    df = pd.read_csv(csv_path)

    # Find title column
    title_col = None
    for candidate in ["title", "headline", "Headline", "Title"]:
        if candidate in df.columns:
            title_col = candidate
            break
    if title_col is None:
        raise ValueError(f"No title column found. Columns: {list(df.columns)}")

    # Find article body column
    text_col = None
    for candidate in ["text", "body", "content", "article"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(f"No article body column found. Columns: {list(df.columns)}")

    # Find label column
    label_col = None
    for candidate in ["label", "Label", "class", "Class"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    # Filter clickbait rows
    unique_labels = df[label_col].unique()
    if "clickbait" in unique_labels:
        df = df[df[label_col] == "clickbait"]
    else:
        df = df[df[label_col] == 1]

    df = df[[text_col, title_col]].dropna()
    pairs = list(zip(df[text_col].tolist(), df[title_col].tolist()))
    print(f"Loaded {len(pairs)} (article, title) pairs from '{csv_path}'.")
    return pairs


def build_vocab(texts: list[str], min_freq: int = 2) -> tuple[dict, dict]:
    """
    Build word2idx and idx2word from a list of text strings.

    Args:
        texts:    List of raw text strings (articles + titles combined).
        min_freq: Minimum word frequency to include in vocabulary.

    Returns:
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
    """
    from collections import Counter

    freq = Counter()
    for t in texts:
        freq.update(tokenize(clean_text(t)))

    word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for word, count in freq.most_common():
        if count >= min_freq and word not in word2idx:
            word2idx[word] = len(word2idx)

    idx2word = {i: w for w, i in word2idx.items()}
    print(f"Vocabulary size: {len(word2idx)}")
    return word2idx, idx2word


def save_vocab(word2idx: dict, idx2word: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(
            {"word2idx": word2idx, "idx2word": {str(k): v for k, v in idx2word.items()}},
            f,
            indent=2,
        )
    print(f"Vocabulary saved to '{vocab_path}'.")


def load_vocab(vocab_path: str) -> tuple[dict, dict]:
    with open(vocab_path, "r") as f:
        data = json.load(f)
    word2idx = data["word2idx"]
    idx2word = {int(k): v for k, v in data["idx2word"].items()}
    return word2idx, idx2word


class ArticleTitleDataset(Dataset):
    """
    PyTorch Dataset of (article_tokens, dec_input_tokens, target_tokens) triples.

    For each (article, title) pair:
      article_tensor:   article padded/truncated to max_article_len
      dec_input_tensor: [<START>, w1, ..., wN] padded to max_title_len  (decoder input)
      target_tensor:    [w1, ..., wN, <END>]   padded to max_title_len  (decoder target)

    During training the decoder sees the ground-truth previous token at each step
    (teacher forcing). dec_input and target are offset by one position, so the model
    learns: given article context + previous token, predict the next token.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        word2idx: dict,
        max_article_len: int = 100,
        max_title_len: int = 20,
    ):
        self.max_article_len = max_article_len
        self.max_title_len = max_title_len
        self.word2idx = word2idx
        self.samples: list[tuple[list[int], list[int], list[int]]] = []

        for article, title in pairs:
            art_ids = self._encode_pad(article, max_article_len)
            dec_input, target = self._encode_title(title, max_title_len)
            self.samples.append((art_ids, dec_input, target))

        print(f"Dataset: {len(self.samples)} samples.")

    def _encode_pad(self, text: str, max_len: int) -> list[int]:
        """Tokenize, truncate, convert to indices, right-pad with PAD."""
        tokens = tokenize(clean_text(text))[:max_len]
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        ids += [PAD_IDX] * (max_len - len(ids))
        return ids

    def _encode_title(self, title: str, max_len: int) -> tuple[list[int], list[int]]:
        """
        Build teacher-forcing (dec_input, target) pair for a title.

        For a title with words [w1, w2, w3] and max_len=6:
          dec_input = [START, w1, w2, w3, PAD, PAD]
          target    = [w1,    w2, w3, END, PAD, PAD]
        """
        tokens = tokenize(clean_text(title))
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        # Reserve one slot: START in dec_input, END in target
        ids = ids[: max_len - 1]

        dec_input = [START_IDX] + ids
        target = ids + [END_IDX]

        pad_len = max_len - len(dec_input)
        dec_input += [PAD_IDX] * pad_len
        target += [PAD_IDX] * (max_len - len(target))
        return dec_input, target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        art, dec_in, tgt = self.samples[idx]
        return (
            torch.tensor(art, dtype=torch.long),
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


def build_dataloaders(
    csv_path: str,
    output_dir: str = "data/",
    val_split: float = 0.1,
    batch_size: int = 64,
    max_article_len: int = 100,
    max_title_len: int = 20,
    min_freq: int = 2,
) -> tuple[DataLoader, DataLoader, dict, dict]:
    """
    Full pipeline: load data -> build vocab -> create DataLoaders.

    Vocab is built from both article bodies and titles in the training split
    so encoder and decoder share the same word2idx.

    Returns:
        train_loader, val_loader, word2idx, idx2word
    """
    pairs = load_article_title_pairs(csv_path)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    n_val = int(len(pairs) * val_split)
    val_pairs = [pairs[i] for i in indices[:n_val]]
    train_pairs = [pairs[i] for i in indices[n_val:]]

    # Build vocab from all training text (articles + titles)
    all_texts = [art for art, _ in train_pairs] + [title for _, title in train_pairs]
    word2idx, idx2word = build_vocab(all_texts, min_freq=min_freq)
    save_vocab(word2idx, idx2word, output_dir)

    train_ds = ArticleTitleDataset(train_pairs, word2idx, max_article_len, max_title_len)
    val_ds = ArticleTitleDataset(val_pairs, word2idx, max_article_len, max_title_len)

    num_workers = 0 if __import__("sys").platform == "win32" else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, word2idx, idx2word


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vocab and inspect dataset.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--max_article_len", type=int, default=100)
    parser.add_argument("--max_title_len", type=int, default=20)
    parser.add_argument("--min_freq", type=int, default=2)
    args = parser.parse_args()

    train_loader, val_loader, word2idx, idx2word = build_dataloaders(
        csv_path=args.data_path,
        output_dir=args.output_dir,
        max_article_len=args.max_article_len,
        max_title_len=args.max_title_len,
        min_freq=args.min_freq,
    )

    for article, dec_input, target in train_loader:
        print(f"Article batch shape:   {article.shape}")
        print(f"Dec input batch shape: {dec_input.shape}")
        print(f"Target batch shape:    {target.shape}")
        sample_art = [idx2word[i.item()] for i in article[0] if i.item() != PAD_IDX]
        sample_tgt = [idx2word[i.item()] for i in target[0] if i.item() != PAD_IDX]
        print(f"Sample article (first 10 tokens): {sample_art[:10]}")
        print(f"Sample target title: {sample_tgt}")
        break
