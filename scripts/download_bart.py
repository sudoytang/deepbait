"""
scripts/download_bart.py
------------------------
Download and cache a pre-trained BART model from HuggingFace.

The model is saved to a local directory so subsequent runs don't need
internet access. Default: facebook/bart-large-cnn (406M params,
already fine-tuned on CNN/DailyMail summarisation).

Usage (from project root):
    uv run python scripts/download_bart.py
    uv run python scripts/download_bart.py --model facebook/bart-base
    uv run python scripts/download_bart.py --output_dir models/bart-large-cnn
"""

import argparse
import os
import time


def human_size(path: str) -> str:
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Download a BART model from HuggingFace.")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-cnn",
        help="HuggingFace model ID (default: facebook/bart-large-cnn).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local directory to save the model. "
             "Defaults to models/<model-name>, e.g. models/bart-large-cnn.",
    )
    args = parser.parse_args()

    model_id = args.model
    output_dir = args.output_dir or os.path.join("models", model_id.split("/")[-1])

    print(f"Model    : {model_id}")
    print(f"Save to  : {output_dir}/")
    print()

    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Directory '{output_dir}' already exists and is not empty.")
        print("Delete it or choose a different --output_dir to re-download.")
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print("Downloading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"  done ({time.time()-t0:.1f}s)")

    print("Downloading model weights (this may take a few minutes)...")
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"  done ({time.time()-t0:.1f}s)")

    print(f"Saving to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    params = sum(p.numel() for p in model.parameters())
    print()
    print("--- Summary ---")
    print(f"  Parameters : {params:,}")
    print(f"  Disk size  : {human_size(output_dir)}")
    print(f"  Location   : {output_dir}/")
    print()
    print("To fine-tune this model on clickbait data, run:")
    print(f"  uv run python scripts/run_bart_finetune.py --model_dir {output_dir}")


if __name__ == "__main__":
    main()
