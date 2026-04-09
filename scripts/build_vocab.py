"""
scripts/build_vocab.py
----------------------
Build a shared vocabulary from the largest available dataset and save it
to a standalone file. All experiments (run_direct.py, run_pretrain_finetune.py,
etc.) load this file via --vocab_from so they all use an identical vocabulary,
making their PPL scores and model sizes directly comparable.

Usage (from project root):
    uv run python scripts/build_vocab.py

    # Quick test with fewer HuggingFace samples:
    uv run python scripts/build_vocab.py --hf_max_samples 50000

    # Custom output path:
    uv run python scripts/build_vocab.py --output checkpoints/my_vocab.pt

What this script uses to build the vocab (in order of data volume):
    1. CNN / DailyMail (287K articles)  -- HuggingFace, downloaded automatically
    2. data/train.csv (all labels, 24K articles)
    3. data/webis17/...  (19K posts, optional)

With min_freq=100 on this corpus the resulting vocabulary is ~25K words,
which is large enough to cover clickbait topics yet small enough to keep
model parameters comparable to Exp 1's natural vocabulary.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import (
    build_vocab,
    clean_text,
    load_article_title_pairs,
    load_hf_dataset,
    load_webis17,
    save_vocab,
    tokenize,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build a shared vocabulary for all experiments."
    )
    parser.add_argument("--data_path", type=str, default="data/train.csv",
                        help="Primary CSV dataset (all labels, not just clickbait).")
    parser.add_argument("--webis17_path", type=str,
                        default="data/webis17/clickbait17-validation-170630",
                        help="Webis-17 directory (set to empty string to skip).")
    parser.add_argument("--hf_dataset", type=str, default="cnn_dailymail:3.0.0",
                        help="HuggingFace dataset (set to empty string to skip).")
    parser.add_argument("--hf_max_samples", type=int, default=None,
                        help="Cap HuggingFace samples for a quick test run.")
    parser.add_argument("--min_freq", type=int, default=100,
                        help="Minimum word frequency to include in vocabulary.")
    parser.add_argument("--output", type=str, default="checkpoints/shared_vocab.pt",
                        help="Where to save the vocabulary file.")
    parser.add_argument("--vocab_dir", type=str, default="data/",
                        help="Directory to also write vocab.json (for inspection).")
    args = parser.parse_args()

    # ----------------------------------------------------------------- Collect texts
    all_texts: list[str] = []

    # 1. Primary CSV (all labels — we want general language coverage)
    pairs = load_article_title_pairs(args.data_path, clickbait_only=False)
    for article, title in pairs:
        all_texts.append(article)
        all_texts.append(title)
    print(f"CSV:      {len(pairs)} pairs  ({len(all_texts)} texts so far)")

    # 2. Webis-17 (all records, regardless of truthMean — we just want vocab coverage)
    if args.webis17_path:
        webis_pairs = load_webis17(args.webis17_path, min_truthmean=0.0)
        for article, title in webis_pairs:
            all_texts.append(article)
            all_texts.append(title)
        print(f"Webis-17: {len(webis_pairs)} pairs  ({len(all_texts)} texts so far)")

    # 3. HuggingFace dataset
    if args.hf_dataset:
        hf_pairs = load_hf_dataset(
            args.hf_dataset,
            split="train",
            max_samples=args.hf_max_samples,
        )
        for article, title in hf_pairs:
            all_texts.append(article)
            all_texts.append(title)
        print(f"HF:       {len(hf_pairs)} pairs  ({len(all_texts)} texts so far)")

    # ----------------------------------------------------------------- Build vocab
    print(f"\nBuilding vocabulary (min_freq={args.min_freq})...")
    word2idx, idx2word = build_vocab(all_texts, min_freq=args.min_freq)

    # ----------------------------------------------------------------- Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"word2idx": word2idx, "idx2word": idx2word}, args.output)
    print(f"Vocabulary saved to '{args.output}'.")

    save_vocab(word2idx, idx2word, args.vocab_dir)

    # ----------------------------------------------------------------- Summary
    print("\n--- Summary ---")
    print(f"  Vocabulary size : {len(word2idx):,}")
    print(f"  Standalone file : {args.output}")
    print(f"  JSON (inspect)  : {os.path.join(args.vocab_dir, 'vocab.json')}")
    print("\nPass --vocab_from to other scripts to use this vocabulary:")
    print(f"  uv run python scripts/run_direct.py --vocab_from {args.output}")
    print(f"  uv run python scripts/run_pretrain_finetune.py --vocab_from {args.output}")


if __name__ == "__main__":
    main()
