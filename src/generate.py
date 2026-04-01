"""
generate.py
-----------
Inference script for the Seq2SeqClickbait model.
Takes an article as input and generates a clickbait headline for it.

Usage:
    python src/generate.py --checkpoint checkpoints/best_model.pt \
                           --article "Scientists discover a new species of frog in the Amazon rainforest." \
                           --temperature 0.8 \
                           --num_headlines 5
"""

import argparse

import torch
import torch.nn.functional as F

from data_processing import (
    END_IDX,
    PAD_IDX,
    START_IDX,
    UNK_IDX,
    clean_text,
    tokenize,
)
from model import ArticleEncoder, ClickbaitDecoder, Seq2SeqClickbait


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[Seq2SeqClickbait, dict, dict, int]:
    """
    Load a trained Seq2SeqClickbait from a checkpoint file.

    Returns:
        model:           Loaded model in eval mode.
        word2idx:        Vocabulary word -> index mapping.
        idx2word:        Vocabulary index -> word mapping.
        max_article_len: Article truncation length used during training.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint["hyperparams"]
    word2idx = checkpoint["word2idx"]
    idx2word = {v: k for k, v in word2idx.items()}

    encoder = ArticleEncoder(
        vocab_size=hp["vocab_size"],
        embed_dim=hp["embed_dim"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
        pad_idx=PAD_IDX,
    )
    decoder = ClickbaitDecoder(
        vocab_size=hp["vocab_size"],
        embed_dim=hp["embed_dim"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
        pad_idx=PAD_IDX,
    )
    model = Seq2SeqClickbait(encoder, decoder).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, word2idx, idx2word, hp["max_article_len"]


def temperature_sample(logits: torch.Tensor, temperature: float) -> int:
    """
    Sample a token index from logits using temperature scaling.

    T < 1.0 -> more deterministic (peaked distribution)
    T = 1.0 -> standard softmax sampling
    T > 1.0 -> more random (flatter distribution)
    """
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def generate_headline(
    model: Seq2SeqClickbait,
    word2idx: dict,
    idx2word: dict,
    article_text: str,
    max_article_len: int = 100,
    max_len: int = 20,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate a single clickbait headline for the given article.

    Args:
        model:           Trained Seq2SeqClickbait in eval mode.
        word2idx:        Vocabulary mapping.
        idx2word:        Reverse vocabulary mapping.
        article_text:    Input article as a plain string.
        max_article_len: Truncation length for the article (must match training).
        max_len:         Maximum number of words to generate.
        temperature:     Sampling temperature.
        device:          Torch device.

    Returns:
        Generated headline string (without special tokens).
    """
    model.eval()

    # Encode the article into a fixed-length token id sequence
    art_tokens = tokenize(clean_text(article_text))[:max_article_len]
    art_ids = [word2idx.get(t, UNK_IDX) for t in art_tokens]
    art_ids += [PAD_IDX] * (max_article_len - len(art_ids))
    article_tensor = torch.tensor([art_ids], dtype=torch.long, device=device)

    generated_words = []

    with torch.no_grad():
        # Encode article -> initial decoder hidden state
        hidden = model.encode(article_tensor)

        current_idx = START_IDX

        for _ in range(max_len):
            token = torch.tensor([[current_idx]], dtype=torch.long, device=device)
            logits, hidden = model.decode_step(token, hidden)
            next_logits = logits[0, 0, :]  # (vocab_size,)

            next_idx = temperature_sample(next_logits, temperature)

            if next_idx == END_IDX:
                break

            word = idx2word.get(next_idx, "<UNK>")
            generated_words.append(word)
            current_idx = next_idx

    special = {"<PAD>", "<UNK>", "<START>", "<END>"}
    return " ".join(w for w in generated_words if w not in special)


def generate_batch(
    model: Seq2SeqClickbait,
    word2idx: dict,
    idx2word: dict,
    article_text: str,
    num_headlines: int = 5,
    max_article_len: int = 100,
    max_len: int = 20,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """Generate multiple headline candidates for the same article."""
    return [
        generate_headline(
            model, word2idx, idx2word, article_text,
            max_article_len, max_len, temperature, device,
        )
        for _ in range(num_headlines)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a clickbait headline for an article.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--article",
        type=str,
        required=True,
        help="Article text to generate a clickbait headline for.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_headlines",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="Maximum number of words in the generated headline.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, word2idx, idx2word, max_article_len = load_model(args.checkpoint, device)

    print(f"\nArticle: {args.article[:200]}{'...' if len(args.article) > 200 else ''}")
    print(f"Generating {args.num_headlines} headline(s) | temperature: {args.temperature}\n")

    headlines = generate_batch(
        model=model,
        word2idx=word2idx,
        idx2word=idx2word,
        article_text=args.article,
        num_headlines=args.num_headlines,
        max_article_len=max_article_len,
        max_len=args.max_len,
        temperature=args.temperature,
        device=device,
    )

    for i, h in enumerate(headlines, 1):
        print(f"  {i}. {h}")
