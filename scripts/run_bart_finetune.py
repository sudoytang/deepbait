"""
scripts/run_bart_finetune.py
-----------------------------
Experiment 3: Fine-tune a pre-trained BART model on clickbait data.

BART (facebook/bart-large-cnn) is already trained on CNN/DailyMail
summarisation, so it can already "read an article and write a short
sentence". We just need a few epochs to shift its style toward clickbait.

Usage (from project root):
    # Default run (bart-large-cnn, 5 epochs, fp16)
    uv run python scripts/run_bart_finetune.py

    # Fewer epochs for a quick test
    uv run python scripts/run_bart_finetune.py --epochs 2

    # Use bart-base instead
    uv run python scripts/run_bart_finetune.py --model_dir models/bart-base

Requirements:
    uv run python scripts/download_bart.py   (run this first)
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import load_article_title_pairs, load_webis17


class ClickbaitSeq2SeqDataset(Dataset):
    """Tokenised (article -> title) pairs ready for BART fine-tuning."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer,
        max_input_len: int = 512,
        max_target_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.samples: list[dict] = []

        for article, title in pairs:
            enc = tokenizer(
                article,
                max_length=max_input_len,
                truncation=True,
                padding=False,
            )
            dec = tokenizer(
                title,
                max_length=max_target_len,
                truncation=True,
                padding=False,
            )
            self.samples.append({
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels":         dec["input_ids"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 — Fine-tune BART on clickbait data."
    )
    parser.add_argument("--model_dir", type=str, default="models/bart-large-cnn",
                        help="Local directory of the downloaded BART model.")
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--webis17_path", type=str,
                        default="data/webis17/clickbait17-validation-170630")
    parser.add_argument("--webis17_min_score", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="checkpoints/exp3_bart")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size. Effective batch = batch_size × grad_accum.")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × this).")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate (3e-5 is typical for BART fine-tuning).")
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="NumPy seed for shuffling before train/validation split; also passed to Trainer as seed.",
    )
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs).")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use fp16 mixed precision (default: True, requires CUDA).")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    args = parser.parse_args()

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    use_fp16 = args.fp16 and torch.cuda.is_available()

    print("=" * 60)
    print("Experiment 3: Fine-tuning BART on clickbait data")
    print(f"  Model:         {args.model_dir}")
    print(f"  Data:          {args.data_path} + {args.webis17_path}")
    print(f"  Split seed:    {args.split_seed}")
    print(f"  Epochs:        {args.epochs}  (early stop patience={args.patience})")
    print(f"  Batch size:    {args.batch_size} × {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"  LR:            {args.lr}")
    print(f"  fp16:          {use_fp16}")
    print(f"  Output:        {args.output_dir}/")
    print("=" * 60)

    # ------------------------------------------------------------------ Data
    pairs = load_article_title_pairs(args.data_path, clickbait_only=True)

    if args.webis17_path:
        webis_pairs = load_webis17(args.webis17_path, min_truthmean=args.webis17_min_score)
        pairs = pairs + webis_pairs
        print(f"Combined: {len(pairs)} (article, title) pairs.")

    np.random.seed(args.split_seed)
    indices = np.random.permutation(len(pairs))
    n_val = int(len(pairs) * args.val_split)
    val_pairs   = [pairs[i] for i in indices[:n_val]]
    train_pairs = [pairs[i] for i in indices[n_val:]]
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # --------------------------------------------------------------- Tokenizer
    print(f"\nLoading tokenizer from '{args.model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    train_ds = ClickbaitSeq2SeqDataset(train_pairs, tokenizer,
                                        args.max_input_len, args.max_target_len)
    val_ds   = ClickbaitSeq2SeqDataset(val_pairs,   tokenizer,
                                        args.max_input_len, args.max_target_len)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)

    # ----------------------------------------------------------------- Model
    print(f"Loading model from '{args.model_dir}'...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # -------------------------------------------------------- Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=args.split_seed,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    # ---------------------------------------------------------------- Train
    print("\nStarting fine-tuning...")
    trainer.train()

    # ------------------------------------------------------------- Save best
    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nBest model saved to '{best_dir}'.")
    print("To generate headlines, run:")
    print(f"  uv run python scripts/generate_bart.py --model_dir {best_dir}")


if __name__ == "__main__":
    main()
