"""
evaluate.py
-----------
Evaluation script for the ClickbaitLSTM model.

1. Computes perplexity on the validation split.
2. Generates 50 diverse headlines across multiple seeds and temperatures.
3. Ranks headlines as "best" and "worst" using simple heuristics.
4. Saves all results to outputs/generated_headlines.txt.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt \
                           --data_path data/train.csv
"""

import argparse
import math
import os

import torch
import torch.nn as nn

from data_processing import PAD_IDX, build_dataloaders
from generate import generate_batch, load_model
from model import ClickbaitLSTM


# ------------------------------------------------------------------ Heuristics

CLICKBAIT_PATTERNS = [
    "you", "your", "will", "this", "things", "why", "how",
    "never", "always", "reasons", "ways", "actually", "really",
    "know", "believe", "told", "secret", "shocking", "amazing",
]


def score_headline(headline: str) -> float:
    """
    Simple quality score for a generated headline.
    Higher is better.

    Criteria:
    - Reasonable word count (5-15 words)
    - Contains clickbait-style vocabulary
    - No excessive repetition
    """
    words = headline.split()
    if len(words) == 0:
        return 0.0

    score = 0.0

    # Length reward: prefer 5-15 words
    if 5 <= len(words) <= 15:
        score += 2.0
    elif len(words) < 3:
        score -= 2.0

    # Clickbait vocabulary presence
    lower_words = [w.lower() for w in words]
    pattern_hits = sum(1 for p in CLICKBAIT_PATTERNS if p in lower_words)
    score += pattern_hits * 0.5

    # Repetition penalty: unique word ratio
    unique_ratio = len(set(lower_words)) / len(lower_words)
    score += unique_ratio * 2.0

    return score


# -------------------------------------------------------------- Perplexity

def compute_perplexity(
    model: ClickbaitLSTM,
    loader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """Compute perplexity on a DataLoader (val or test split)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, targets)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return math.exp(avg_loss)


# --------------------------------------------------------------- Main Eval

SEED_PHRASES = [
    "",                       # unconditional
    "10 things",
    "you won't believe",
    "this is why",
    "the real reason",
    "what happens when",
    "we asked",
    "here's what",
    "the truth about",
    "why you should",
]

TEMPERATURES = [0.5, 0.8, 1.0, 1.2]


def run_evaluation(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, word2idx, idx2word = load_model(args.checkpoint, device)

    # ---------------------------------------------------------- Perplexity
    print("\nComputing validation perplexity...")
    _, val_loader, _, _ = build_dataloaders(
        csv_path=args.data_path,
        output_dir=args.output_dir,
        val_split=0.1,
        batch_size=64,
        max_seq_len=args.max_seq_len,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    ppl = compute_perplexity(model, val_loader, criterion, device)
    print(f"Validation Perplexity: {ppl:.2f}")

    # ------------------------------------------------ Generate 50 headlines
    print("\nGenerating headlines across seeds and temperatures...")
    all_headlines: list[dict] = []

    headlines_per_combo = max(1, 50 // (len(SEED_PHRASES) * len(TEMPERATURES)))

    for seed in SEED_PHRASES:
        for temp in TEMPERATURES:
            batch = generate_batch(
                model=model,
                word2idx=word2idx,
                idx2word=idx2word,
                seed_phrase=seed,
                num_headlines=headlines_per_combo,
                max_len=args.max_len,
                temperature=temp,
                device=device,
            )
            for h in batch:
                all_headlines.append(
                    {
                        "headline": h,
                        "seed": seed if seed else "(unconditional)",
                        "temperature": temp,
                        "score": score_headline(h),
                    }
                )

    # Deduplicate
    seen = set()
    unique_headlines = []
    for item in all_headlines:
        if item["headline"] not in seen:
            seen.add(item["headline"])
            unique_headlines.append(item)

    unique_headlines.sort(key=lambda x: x["score"], reverse=True)

    best = unique_headlines[:10]
    worst = unique_headlines[-10:]

    # ---------------------------------------------------------- Print & Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "generated_headlines.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("CLICKBAIT HEADLINE GENERATOR — EVALUATION RESULTS")
    lines.append("=" * 70)
    lines.append(f"\nValidation Perplexity: {ppl:.2f}\n")

    lines.append("-" * 70)
    lines.append("TOP 10 BEST HEADLINES")
    lines.append("-" * 70)
    for i, item in enumerate(best, 1):
        lines.append(
            f"  {i:2d}. {item['headline']}\n"
            f"      [seed: {item['seed']} | T={item['temperature']} | score={item['score']:.2f}]"
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append("10 WORST HEADLINES (lowest quality score)")
    lines.append("-" * 70)
    for i, item in enumerate(worst, 1):
        lines.append(
            f"  {i:2d}. {item['headline']}\n"
            f"      [seed: {item['seed']} | T={item['temperature']} | score={item['score']:.2f}]"
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append(f"ALL {len(unique_headlines)} UNIQUE GENERATED HEADLINES")
    lines.append("-" * 70)
    for i, item in enumerate(unique_headlines, 1):
        lines.append(
            f"  {i:3d}. {item['headline']}\n"
            f"       [seed: {item['seed']} | T={item['temperature']}]"
        )

    output_text = "\n".join(lines)
    print(output_text)

    with open(out_path, "w") as f:
        f.write(output_text)

    print(f"\nResults saved to '{out_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the ClickbaitLSTM model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the original CSV (used to rebuild val split)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Directory to save generated headlines",
    )
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=20, help="Max words per generated headline")
    args = parser.parse_args()

    run_evaluation(args)
