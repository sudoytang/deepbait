"""
scripts/generate_bart.py
-------------------------
Generate clickbait headlines with the fine-tuned BART model.

Usage:
    uv run python scripts/generate_bart.py --model_dir checkpoints/exp3_bart/best_model

    # Interactive mode (prompts for article text)
    uv run python scripts/generate_bart.py --model_dir checkpoints/exp3_bart/best_model --interactive

    # From file
    uv run python scripts/generate_bart.py --model_dir checkpoints/exp3_bart/best_model --article_file article.txt
"""

import argparse
import sys


def generate(model, tokenizer, article: str, num_beams: int = 4,
             max_input_len: int = 512, max_output_len: int = 64) -> str:
    import torch
    device = next(model.parameters()).device
    inputs = tokenizer(
        article,
        return_tensors="pt",
        max_length=max_input_len,
        truncation=True,
        return_token_type_ids=False,
    ).to(device)

    # Ban social-media noise tokens (#, @, and their byte-pair variants)
    noise_strings = ["#", "@", " #", " @"]
    bad_words_ids = [tokenizer.encode(s, add_special_tokens=False) for s in noise_strings]
    bad_words_ids = [ids for ids in bad_words_ids if ids]

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=num_beams,
        min_length=1,
        max_new_tokens=max_output_len,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_ids,
        early_stopping=True,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.split("\n")[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate clickbait headlines with BART.")
    parser.add_argument("--model_dir", type=str,
                        default="checkpoints/exp3_bart/best_model",
                        help="Path to fine-tuned (or base) BART model directory.")
    parser.add_argument("--article", type=str, default=None,
                        help="Article text as a CLI argument.")
    parser.add_argument("--article_file", type=str, default=None,
                        help="Path to a text file containing the article.")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive prompt loop.")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_output_len", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from '{args.model_dir}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.generation_config.max_length = None
    model.generation_config.min_length = 1
    model.eval()
    print("Model loaded.\n")

    def run(text: str):
        headline = generate(model, tokenizer, text, args.num_beams,
                            args.max_input_len, args.max_output_len)
        print(f"Headline: {headline}")

    if args.article:
        run(args.article)
    elif args.article_file:
        with open(args.article_file, "r", encoding="utf-8") as f:
            run(f.read().strip())
    elif args.interactive:
        print("Enter article text (blank line to submit, 'quit' to exit):\n")
        while True:
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    sys.exit(0)
                if line.strip().lower() == "quit":
                    sys.exit(0)
                if line == "":
                    break
                lines.append(line)
            if lines:
                run(" ".join(lines))
                print()
    else:
        # Demo with a hardcoded example
        demo = (
            "Scientists at MIT have developed a new artificial intelligence system "
            "that can diagnose certain types of cancer from blood samples with "
            "accuracy comparable to experienced physicians. The research, published "
            "in Nature Medicine, used deep learning to analyse protein biomarkers "
            "in over 10,000 patient samples."
        )
        print("Demo article:")
        print(demo)
        print()
        run(demo)


if __name__ == "__main__":
    main()
