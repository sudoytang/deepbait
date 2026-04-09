"""
train.py
--------
Training loop for the Seq2SeqClickbait model.

Usage:
    python src/train.py --data_path data/train.csv --epochs 20 --batch_size 64
"""

import argparse
import json
import math
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_processing import PAD_IDX, build_dataloaders, load_vocab
from model import ArticleEncoder, ClickbaitDecoder, Seq2SeqClickbait


def train_epoch(
    model: Seq2SeqClickbait,
    loader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float = 1.0,
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    vocab_size = model.decoder.vocab_size

    for article, dec_input, target in loader:
        article   = article.to(device)    # (batch, article_len)
        dec_input = dec_input.to(device)  # (batch, title_len)
        target    = target.to(device)     # (batch, title_len)

        optimizer.zero_grad()

        # Full teacher-forcing forward pass
        logits = model(article, dec_input)  # (batch, title_len, vocab_size)

        # Flatten for CrossEntropyLoss; PAD positions are ignored by criterion
        loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(
    model: Seq2SeqClickbait,
    loader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """Run evaluation, return average loss."""
    model.eval()
    total_loss = 0.0
    vocab_size = model.decoder.vocab_size

    with torch.no_grad():
        for article, dec_input, target in loader:
            article   = article.to(device)
            dec_input = dec_input.to(device)
            target    = target.to(device)

            logits = model(article, dec_input)
            loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(
    model: Seq2SeqClickbait,
    optimizer,
    epoch: int,
    val_loss: float,
    word2idx: dict,
    hyperparams: dict,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "word2idx": word2idx,
            "hyperparams": hyperparams,
        },
        save_path,
    )


def plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    train_ppl = [math.exp(l) for l in train_losses]
    val_ppl = [math.exp(l) for l in val_losses]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_ppl, label="Train Perplexity")
    axes[1].plot(epochs, val_ppl, label="Val Perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Training and Validation Perplexity")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss/perplexity curves saved to '{save_path}'.")


def load_checkpoint_weights(
    checkpoint_path: str,
    model: "Seq2SeqClickbait",
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """
    Load model (and optionally optimizer) state from a checkpoint.

    Returns the epoch number stored in the checkpoint.
    When fine-tuning, pass optimizer=None to start with a fresh optimizer
    (prevents inheriting the old learning-rate state).
    """
    print(f"Resuming from checkpoint: '{checkpoint_path}'")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train/val split_seed (NumPy): {args.split_seed}")

    # ------------------------------------------------------------------ Data
    # Priority for vocabulary (highest to lowest):
    #   1. --vocab_from  : standalone vocab file built by build_vocab.py
    #   2. --resume_checkpoint : reuse the vocab embedded in a training checkpoint
    #   3. (default) build vocab fresh from the current training data
    preset_vocab = None
    if getattr(args, "vocab_from", None):
        vf = torch.load(args.vocab_from, map_location="cpu")
        w2i = vf["word2idx"]
        i2w = vf.get("idx2word") or {v: k for k, v in w2i.items()}
        if isinstance(next(iter(i2w)), str):
            i2w = {int(k): v for k, v in i2w.items()}
        preset_vocab = (w2i, i2w)
        print(f"Loaded shared vocab from '{args.vocab_from}' (size={len(w2i)}).")
    elif args.resume_checkpoint:
        ckpt_for_vocab = torch.load(args.resume_checkpoint, map_location="cpu")
        w2i = ckpt_for_vocab["word2idx"]
        i2w = {v: k for k, v in w2i.items()}
        preset_vocab = (w2i, i2w)
        print(f"Loaded vocab from checkpoint (size={len(w2i)}).")

    train_loader, val_loader, word2idx, idx2word = build_dataloaders(
        csv_path=args.data_path,
        output_dir=args.output_dir,
        val_split=0.1,
        batch_size=args.batch_size,
        max_article_len=args.max_article_len,
        max_title_len=args.max_title_len,
        min_freq=args.min_freq,
        clickbait_only=not args.no_clickbait_filter,
        webis17_path=args.webis17_path,
        webis17_min_score=args.webis17_min_score,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_article_col=args.hf_article_col,
        hf_title_col=args.hf_title_col,
        hf_max_samples=args.hf_max_samples,
        preset_vocab=preset_vocab,
        split_seed=args.split_seed,
    )

    vocab_size = len(word2idx)
    hyperparams = {
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "max_article_len": args.max_article_len,
        "max_title_len": args.max_title_len,
        "split_seed": args.split_seed,
    }

    # ----------------------------------------------------------------- Model
    encoder = ArticleEncoder(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
    )
    decoder = ClickbaitDecoder(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
    )
    model = Seq2SeqClickbait(encoder, decoder).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # ------------------------------------------------- Optional: resume weights
    if args.resume_checkpoint:
        # For fine-tuning: load model weights only, keep fresh optimizer with new lr
        load_checkpoint_weights(args.resume_checkpoint, model, optimizer=None, device=device)

    # --------------------------------------------------------------- Training
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    no_improve_count = 0
    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pt")
    patience = args.early_stopping_patience

    print(f"\nStarting training for {args.epochs} epochs...")
    if patience > 0:
        print(f"Early stopping patience: {patience} epochs")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} (PPL {train_ppl:.1f}) | "
            f"Val Loss: {val_loss:.4f} (PPL {val_ppl:.1f}) | "
            f"Time: {elapsed:.1f}s"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            save_checkpoint(model, optimizer, epoch, val_loss, word2idx, hyperparams, best_ckpt)
            print(f"  -> New best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            no_improve_count += 1
            if patience > 0:
                print(f"  (no improvement {no_improve_count}/{patience})")
            if patience > 0 and no_improve_count >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    final_ckpt = os.path.join(args.checkpoint_dir, "final_model.pt")
    save_checkpoint(model, optimizer, epoch, val_losses[-1], word2idx, hyperparams, final_ckpt)

    curve_path = os.path.join(args.checkpoint_dir, "loss_curve.png")
    plot_curves(train_losses, val_losses, curve_path)

    history = {"train_loss": train_losses, "val_loss": val_losses}
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} (PPL {math.exp(best_val_loss):.1f})")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Seq2SeqClickbait model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--output_dir", type=str, default="data/", help="Directory to save vocab")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_article_len", type=int, default=100)
    parser.add_argument("--max_title_len", type=int, default=20)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--no_clickbait_filter", action="store_true",
                        help="Include all articles regardless of label (for pre-training).")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load weights from before training.")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop training if val loss does not improve for N epochs. 0 = disabled.")
    parser.add_argument("--vocab_from", type=str, default=None,
                        help="Path to a standalone vocab file built by scripts/build_vocab.py. "
                             "Overrides vocab built from training data.")
    parser.add_argument("--webis17_path", type=str, default=None,
                        help="Optional path to Webis-Clickbait-17 directory or JSONL file.")
    parser.add_argument("--webis17_min_score", type=float, default=0.5,
                        help="Minimum truthMean to include from Webis-17.")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset name, e.g. 'cnn_dailymail:3.0.0' or 'cc_news'.")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_article_col", type=str, default=None,
                        help="Article column name (auto-detected for known datasets).")
    parser.add_argument("--hf_title_col", type=str, default=None,
                        help="Title column name (auto-detected for known datasets).")
    parser.add_argument("--hf_max_samples", type=int, default=None,
                        help="Max number of samples to load from HuggingFace dataset.")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="NumPy seed for shuffling before train/validation split.",
    )
    args = parser.parse_args()

    train(args)