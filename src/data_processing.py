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


def load_article_title_pairs(
    csv_path: str,
    clickbait_only: bool = True,
) -> list[tuple[str, str]]:
    """
    Load the dataset and return (article_text, title) pairs.

    Expected CSV columns: a title column, an article body column, and a label column.

    Args:
        csv_path:       Path to the CSV file.
        clickbait_only: If True (default), only return rows labelled as clickbait.
                        If False, return all rows regardless of label.
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

    if clickbait_only:
        unique_labels = df[label_col].unique()
        if "clickbait" in unique_labels:
            df = df[df[label_col] == "clickbait"]
        else:
            df = df[df[label_col] == 1]

    df = df[[text_col, title_col]].dropna()
    pairs = list(zip(df[text_col].tolist(), df[title_col].tolist()))
    print(f"Loaded {len(pairs)} (article, title) pairs from '{csv_path}'.")
    return pairs


def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    article_col: str | None = None,
    title_col: str | None = None,
    max_samples: int | None = None,
) -> list[tuple[str, str]]:
    """
    Load (article_text, title) pairs from a HuggingFace dataset.

    Built-in column mappings (auto-detected by dataset_name):
      - "cnn_dailymail"  : article="article",   title="highlights"
      - "cc_news"        : article="text",       title="title"
      - "ag_news"        : article="text",       title="label" (skipped, no title col)
      - anything else    : pass article_col and title_col explicitly

    Args:
        dataset_name:  HuggingFace dataset identifier, e.g. "cnn_dailymail" or
                       "cc_news".  For datasets with config names append it with
                       a colon, e.g. "cnn_dailymail:3.0.0".
        split:         Dataset split to load, e.g. "train", "train+validation".
        article_col:   Column name for the article body (overrides auto-detect).
        title_col:     Column name for the title (overrides auto-detect).
        max_samples:   If set, only use the first N samples (useful for quick tests).

    Returns:
        List of (article_text, title) string tuples.

    Examples:
        load_hf_dataset("cnn_dailymail:3.0.0")
        load_hf_dataset("cc_news", max_samples=50000)
        load_hf_dataset("my_org/my_dataset", article_col="body", title_col="headline")
    """
    from datasets import load_dataset as _load_dataset

    # Built-in column name mappings
    _KNOWN = {
        "cnn_dailymail": ("article", "highlights"),
        "cc_news":       ("text",    "title"),
    }

    name_key = dataset_name.split(":")[0]
    if article_col is None or title_col is None:
        known = _KNOWN.get(name_key)
        if known is None:
            raise ValueError(
                f"Unknown dataset '{name_key}'. "
                f"Please pass article_col and title_col explicitly. "
                f"Known datasets: {list(_KNOWN)}"
            )
        article_col = article_col or known[0]
        title_col   = title_col   or known[1]

    # Parse optional config name ("cnn_dailymail:3.0.0" -> name="cnn_dailymail", config="3.0.0")
    parts = dataset_name.split(":", 1)
    load_kwargs: dict = {"path": parts[0]}
    if len(parts) == 2:
        load_kwargs["name"] = parts[1]

    print(f"Loading HuggingFace dataset '{dataset_name}' (split='{split}')...")
    ds = _load_dataset(**load_kwargs, split=split, trust_remote_code=False)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    pairs: list[tuple[str, str]] = []
    for row in ds:
        article = str(row.get(article_col) or "").strip()
        title   = str(row.get(title_col)   or "").strip()
        if article and title:
            pairs.append((article, title))

    print(f"Loaded {len(pairs)} (article, title) pairs from '{dataset_name}'.")
    return pairs


def load_webis17(
    path: str,
    min_truthmean: float = 0.5,
) -> list[tuple[str, str]]:
    """
    Load (article_text, title) pairs from the Webis-Clickbait-17 dataset.

    ``path`` can be:
    - A **directory** that contains ``instances.jsonl`` and optionally
      ``truth.jsonl``.  This is the standard layout produced when you unzip the
      official ``clickbait17-train-*.zip`` archives from Zenodo.
    - A **single JSONL file** where each line already has both content fields
      (``postText``, ``targetParagraphs``) and the truth field (``truthMean``).

    Download from: https://zenodo.org/records/5530410
    Only download the JSON files, NOT the 96 GB WARC archives.

    Fields used:
      - instances.jsonl  ``postText``         list[str]  -> title (first element)
      - instances.jsonl  ``targetParagraphs`` list[str]  -> article body
      - truth.jsonl      ``truthMean``        float      -> clickbait score [0,1]
    """
    import json as _json

    instances_path: str
    truth_path: str | None = None

    if os.path.isdir(path):
        instances_path = os.path.join(path, "instances.jsonl")
        candidate_truth = os.path.join(path, "truth.jsonl")
        if os.path.exists(candidate_truth):
            truth_path = candidate_truth
    else:
        instances_path = path

    # Build id -> truthMean lookup from the separate truth file (if present)
    truth_map: dict[str, float] = {}
    if truth_path is not None:
        with open(truth_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if "id" in rec and "truthMean" in rec:
                    truth_map[rec["id"]] = float(rec["truthMean"])

    pairs: list[tuple[str, str]] = []
    skipped = 0

    with open(instances_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = _json.loads(line)
            except _json.JSONDecodeError:
                continue

            # Determine truthMean: prefer truth_map, fall back to inline field
            rid = record.get("id", "")
            if rid in truth_map:
                truth = truth_map[rid]
            else:
                truth = record.get("truthMean")

            if truth is not None and float(truth) < min_truthmean:
                skipped += 1
                continue

            post_text = record.get("postText") or []
            title = post_text[0].strip() if post_text else ""
            if not title:
                continue

            paragraphs = record.get("targetParagraphs") or []
            article = " ".join(p.strip() for p in paragraphs if p.strip())
            if not article:
                continue

            pairs.append((article, title))

    print(
        f"Loaded {len(pairs)} pairs from Webis-17 (skipped {skipped} low-score records)."
    )
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
    clickbait_only: bool = True,
    webis17_path: str | None = None,
    webis17_min_score: float = 0.5,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    hf_article_col: str | None = None,
    hf_title_col: str | None = None,
    hf_max_samples: int | None = None,
    preset_vocab: tuple[dict, dict] | None = None,
    split_seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict, dict]:
    """
    Full pipeline: load data -> build vocab -> create DataLoaders.

    Vocab is built from both article bodies and titles in the training split
    so encoder and decoder share the same word2idx.

    Args:
        csv_path:         Path to the primary CSV dataset.
        clickbait_only:   If True (default), only use clickbait-labelled rows from
                          csv_path. Set to False for pre-training on all articles.
        webis17_path:     Optional path to Webis-Clickbait-17 directory or JSONL.
        webis17_min_score: Minimum truthMean to include from Webis-17 (default 0.5).
        hf_dataset:       Optional HuggingFace dataset name, e.g. "cnn_dailymail:3.0.0"
                          or "cc_news". Pairs are merged with csv_path pairs.
        hf_split:         HuggingFace dataset split (default "train").
        hf_article_col:   Override article column name (auto-detected for known datasets).
        hf_title_col:     Override title column name (auto-detected for known datasets).
        hf_max_samples:   Cap the number of HuggingFace samples loaded.
        preset_vocab:     Optional (word2idx, idx2word) tuple to reuse an existing
                          vocabulary instead of building a new one from the data.
                          Use this in fine-tune stage to share Stage 1's vocab.
        split_seed:       NumPy seed for shuffling indices before train/val split.

    Returns:
        train_loader, val_loader, word2idx, idx2word
    """
    pairs = load_article_title_pairs(csv_path, clickbait_only=clickbait_only)

    if webis17_path is not None:
        webis_pairs = load_webis17(webis17_path, min_truthmean=webis17_min_score)
        pairs = pairs + webis_pairs
        print(f"After Webis-17: {len(pairs)} total (article, title) pairs.")

    if hf_dataset is not None:
        hf_pairs = load_hf_dataset(
            hf_dataset,
            split=hf_split,
            article_col=hf_article_col,
            title_col=hf_title_col,
            max_samples=hf_max_samples,
        )
        pairs = pairs + hf_pairs
        print(f"After HuggingFace dataset: {len(pairs)} total (article, title) pairs.")

    # Shuffle and split
    np.random.seed(split_seed)
    indices = np.random.permutation(len(pairs))
    n_val = int(len(pairs) * val_split)
    val_pairs = [pairs[i] for i in indices[:n_val]]
    train_pairs = [pairs[i] for i in indices[n_val:]]

    if preset_vocab is not None:
        word2idx, idx2word = preset_vocab
        print(f"Using preset vocabulary (size={len(word2idx)}).")
    else:
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
    parser.add_argument("--webis17_path", type=str, default=None,
                        help="Optional path to Webis-Clickbait-17 directory or JSONL file.")
    parser.add_argument("--webis17_min_score", type=float, default=0.5)
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset name, e.g. 'cnn_dailymail:3.0.0' or 'cc_news'.")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_article_col", type=str, default=None)
    parser.add_argument("--hf_title_col", type=str, default=None)
    parser.add_argument("--hf_max_samples", type=int, default=None)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="NumPy seed for train/validation shuffle before split.",
    )
    args = parser.parse_args()

    train_loader, val_loader, word2idx, idx2word = build_dataloaders(
        csv_path=args.data_path,
        output_dir=args.output_dir,
        max_article_len=args.max_article_len,
        max_title_len=args.max_title_len,
        min_freq=args.min_freq,
        webis17_path=args.webis17_path,
        webis17_min_score=args.webis17_min_score,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_article_col=args.hf_article_col,
        hf_title_col=args.hf_title_col,
        hf_max_samples=args.hf_max_samples,
        split_seed=args.split_seed,
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
