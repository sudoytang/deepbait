"""
scripts/run_direct.py
---------------------
Experiment 1: Train directly on clickbait data until val loss stops improving.

Usage (from project root):
    uv run python scripts/run_direct.py
    uv run python scripts/run_direct.py --epochs 100 --patience 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train as _train_module


def main():
    parser = argparse.ArgumentParser(
        description="Exp 1 — Direct clickbait training with early stopping."
    )
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--valid_path", type=str, default="data/valid.csv",
                        help="Extra CSV to merge into training data (same format as data_path).")
    parser.add_argument("--webis17_path", type=str,
                        default="data/webis17/clickbait17-validation-170630")
    parser.add_argument("--webis17_min_score", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/exp1_direct")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (early stopping will usually trigger earlier).")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without val improvement).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--vocab_from", type=str,
                        default="checkpoints/shared_vocab.pt",
                        help="Shared vocabulary file built by scripts/build_vocab.py. "
                             "All experiments should use the same file for fair comparison.")
    parser.add_argument("--min_freq", type=int, default=10,
                        help="Min word frequency used only when --vocab_from is not found.")
    parser.add_argument("--max_article_len", type=int, default=100)
    parser.add_argument("--max_title_len", type=int, default=20)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="NumPy seed for shuffling before train/validation split.",
    )
    args = parser.parse_args()

    # Build the args namespace expected by train.train()
    import types
    # Use shared vocab if the file exists; fall back to building from data
    vocab_from = args.vocab_from if os.path.exists(args.vocab_from) else None
    if vocab_from:
        print(f"Using shared vocabulary: {vocab_from}")
    else:
        print(f"Shared vocab not found at '{args.vocab_from}'. "
              f"Run scripts/build_vocab.py first for a fair comparison. "
              f"Falling back to building vocab from training data (min_freq={args.min_freq}).")

    train_args = types.SimpleNamespace(
        data_path=args.data_path,
        output_dir="data/",
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_article_len=args.max_article_len,
        max_title_len=args.max_title_len,
        min_freq=args.min_freq,
        # clickbait only — this is the key setting for Exp 1
        no_clickbait_filter=False,
        vocab_from=vocab_from,
        webis17_path=args.webis17_path,
        webis17_min_score=args.webis17_min_score,
        # no HuggingFace dataset
        hf_dataset=None,
        hf_split="train",
        hf_article_col=None,
        hf_title_col=None,
        hf_max_samples=None,
        # no resume
        resume_checkpoint=None,
        early_stopping_patience=args.patience,
        split_seed=args.split_seed,
    )

    print("=" * 60)
    print("Experiment 1: Direct clickbait training")
    print(f"  CSV data:      {args.data_path}")
    print(f"  Webis-17:      {args.webis17_path} (min_score={args.webis17_min_score})")
    print(f"  Split seed:    {args.split_seed}")
    print(f"  Checkpoint:    {args.checkpoint_dir}/")
    print(f"  Early stop:    patience={args.patience}")
    print("=" * 60)

    _train_module.train(train_args)


if __name__ == "__main__":
    main()