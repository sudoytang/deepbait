"""
scripts/run_pretrain_finetune.py
---------------------------------
Experiment 2: Two-stage training.

  Stage 1 — Pre-train on a large general news corpus (no clickbait filter).
             The model learns to read articles and generate any kind of title.

  Stage 2 — Fine-tune on clickbait-only data with a much smaller learning rate.
             The model "shifts" its title style toward clickbait.

Usage (from project root):
    uv run python scripts/run_pretrain_finetune.py

    # Smaller HuggingFace sample for a quick test:
    uv run python scripts/run_pretrain_finetune.py --hf_max_samples 30000

    # Custom stages:
    uv run python scripts/run_pretrain_finetune.py \\
        --pretrain_epochs 10 \\
        --finetune_epochs 30 \\
        --finetune_lr 1e-4
"""

import argparse
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train as _train_module


def make_args(overrides: dict) -> types.SimpleNamespace:
    """Build a SimpleNamespace with sensible defaults, then apply overrides."""
    defaults = dict(
        data_path="data/train.csv",
        output_dir="data/",
        checkpoint_dir="checkpoints/exp2_pretrain",
        epochs=10,
        batch_size=64,
        lr=1e-3,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        max_article_len=100,
        max_title_len=20,
        min_freq=10,
        no_clickbait_filter=False,
        webis17_path=None,
        webis17_min_score=0.5,
        hf_dataset=None,
        hf_split="train",
        hf_article_col=None,
        hf_title_col=None,
        hf_max_samples=None,
        resume_checkpoint=None,
        early_stopping_patience=0,
        split_seed=42,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 2 — Pre-train on general news, then fine-tune on clickbait."
    )
    # Data
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--webis17_path", type=str,
                        default="data/webis17/clickbait17-validation-170630")
    parser.add_argument("--webis17_min_score", type=float, default=0.5)
    parser.add_argument("--hf_dataset", type=str, default="cnn_dailymail:3.0.0",
                        help="HuggingFace dataset for pre-training (default: cnn_dailymail:3.0.0).")
    parser.add_argument("--hf_max_samples", type=int, default=None,
                        help="Limit HuggingFace samples, e.g. 50000 for a quick test.")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="NumPy seed for shuffling before train/validation split (both stages).",
    )

    # Stage 1
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_dir", type=str, default="checkpoints/exp2_pretrain")

    # Stage 2
    parser.add_argument("--finetune_epochs", type=int, default=30)
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="Fine-tune LR should be ~10x smaller than pre-train LR.")
    parser.add_argument("--finetune_patience", type=int, default=8,
                        help="Early stopping patience for fine-tune stage.")
    parser.add_argument("--finetune_dir", type=str, default="checkpoints/exp2_finetune")

    # Shared hyperparams
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--vocab_from", type=str,
                        default="checkpoints/shared_vocab.pt",
                        help="Pre-built shared vocab file from scripts/build_vocab.py. "
                             "If found, both Stage 1 and Stage 2 use it. "
                             "If not found, Stage 1 builds a new vocab and Stage 2 reuses it.")
    parser.add_argument("--min_freq", type=int, default=100,
                        help="Min word frequency used only when --vocab_from is not found.")
    parser.add_argument("--max_article_len", type=int, default=100)
    parser.add_argument("--max_title_len", type=int, default=20)
    args = parser.parse_args()

    pretrain_ckpt = os.path.join(args.pretrain_dir, "best_model.pt")

    # Use shared vocab if available; Stage 2 will inherit it from Stage 1 checkpoint
    vocab_from = args.vocab_from if os.path.exists(args.vocab_from) else None
    if vocab_from:
        print(f"Using shared vocabulary: {vocab_from}")
    else:
        print(f"Shared vocab not found at '{args.vocab_from}'. "
              f"Stage 1 will build vocab (min_freq={args.min_freq}); "
              f"Stage 2 will reuse it from the Stage 1 checkpoint.")

    # ---------------------------------------------------------------- Stage 1
    print("\n" + "=" * 60)
    print("Stage 1: Pre-training on general news")
    print(f"  CSV (all labels): {args.data_path}")
    print(f"  HuggingFace:      {args.hf_dataset}"
          + (f" (max {args.hf_max_samples} samples)" if args.hf_max_samples else ""))
    print(f"  Epochs:           {args.pretrain_epochs}")
    print(f"  LR:               {args.pretrain_lr}")
    print(f"  Output:           {args.pretrain_dir}/")
    print("=" * 60)

    stage1_args = make_args(dict(
        data_path=args.data_path,
        checkpoint_dir=args.pretrain_dir,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.3,
        max_article_len=args.max_article_len,
        max_title_len=args.max_title_len,
        min_freq=args.min_freq,
        vocab_from=vocab_from,
        no_clickbait_filter=True,
        hf_dataset=args.hf_dataset,
        hf_max_samples=args.hf_max_samples,
        webis17_path=None,
        split_seed=args.split_seed,
    ))

    _train_module.train(stage1_args)

    # ---------------------------------------------------------------- Stage 2
    print("\n" + "=" * 60)
    print("Stage 2: Fine-tuning on clickbait data")
    print(f"  Weights from:     {pretrain_ckpt}")
    print(f"  CSV (clickbait):  {args.data_path}")
    print(f"  Webis-17:         {args.webis17_path}")
    print(f"  LR:               {args.finetune_lr}  (reduced from {args.pretrain_lr})")
    print(f"  Epochs:           {args.finetune_epochs}  (early stop patience={args.finetune_patience})")
    print(f"  Output:           {args.finetune_dir}/")
    print("=" * 60)

    stage2_args = make_args(dict(
        data_path=args.data_path,
        checkpoint_dir=args.finetune_dir,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.4,
        max_article_len=args.max_article_len,
        max_title_len=args.max_title_len,
        min_freq=args.min_freq,
        # vocab_from not needed: resume_checkpoint carries the vocab automatically
        vocab_from=None,
        no_clickbait_filter=False,
        webis17_path=args.webis17_path,
        webis17_min_score=args.webis17_min_score,
        resume_checkpoint=pretrain_ckpt,
        early_stopping_patience=args.finetune_patience,
        split_seed=args.split_seed,
    ))

    _train_module.train(stage2_args)

    print("\nAll done.")
    print(f"  Pre-train best checkpoint : {pretrain_ckpt}")
    print(f"  Fine-tune best checkpoint : {os.path.join(args.finetune_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
