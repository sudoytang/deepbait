"""
model.py
--------
Seq2SeqClickbait: ArticleEncoder + ClickbaitDecoder for article-to-title generation.

Architecture:
    Article tokens
        -> ArticleEncoder  (Embedding -> LSTM -> final hidden state)
        -> initial hidden state for decoder
        -> ClickbaitDecoder (Embedding -> LSTM -> Linear)
        -> logits over vocabulary at each title position
"""

import torch
import torch.nn as nn


class ArticleEncoder(nn.Module):
    """
    LSTM encoder that reads an article and produces a context hidden state
    to initialise the decoder.

    Args:
        vocab_size:  Size of the shared vocabulary.
        embed_dim:   Dimensionality of word embeddings.
        hidden_dim:  Number of hidden units in each LSTM layer.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout probability.
        pad_idx:     Index of the <PAD> token.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Article token indices, shape (batch, article_len).

        Returns:
            (h_n, c_n): Final LSTM hidden state, each (num_layers, batch, hidden_dim).
                        Passed directly as initial hidden state to the decoder.
        """
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.lstm(embedded)
        return hidden  # (h_n, c_n)


class ClickbaitDecoder(nn.Module):
    """
    LSTM decoder that generates a clickbait title token by token,
    initialised from the encoder's hidden state.

    Args:
        vocab_size:  Size of the shared vocabulary.
        embed_dim:   Dimensionality of word embeddings.
        hidden_dim:  Number of hidden units in each LSTM layer.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout probability.
        pad_idx:     Index of the <PAD> token.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:      Token indices, shape (batch, seq_len).
            hidden: LSTM state (h, c) to initialise from (typically from encoder).
                    If None, PyTorch initialises to zeros.

        Returns:
            logits: Raw scores, shape (batch, seq_len, vocab_size).
            hidden: Updated (h_n, c_n) for stateful autoregressive decoding.
        """
        embedded = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits, hidden


class Seq2SeqClickbait(nn.Module):
    """
    Encoder-Decoder model for article -> clickbait headline generation.

    The encoder reads the full article and its final hidden state initialises
    the decoder, which generates the headline token by token.

    Encoder and decoder share the same vocabulary (word2idx), embed_dim,
    hidden_dim, and num_layers so hidden states are directly compatible.

    Training uses teacher forcing: the decoder receives the ground-truth
    previous token at each step rather than its own prediction.

    Inference is autoregressive: use encode() once, then call decode_step()
    one token at a time.
    """

    def __init__(self, encoder: ArticleEncoder, decoder: ClickbaitDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, article: torch.Tensor, dec_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass with teacher forcing (training).

        Args:
            article:   Article token indices, (batch, article_len).
            dec_input: Decoder input tokens, (batch, title_len).
                       Expected format: [<START>, w1, w2, ..., wN, <PAD>, ...]

        Returns:
            logits: (batch, title_len, vocab_size)
        """
        hidden = self.encoder(article)
        logits, _ = self.decoder(dec_input, hidden)
        return logits

    # ------------------------------------------------------------------
    # Helpers for autoregressive inference (used by generate.py)
    # ------------------------------------------------------------------

    def encode(
        self, article: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an article; return (h_n, c_n) to seed the decoder."""
        return self.encoder(article)

    def decode_step(
        self,
        token: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decoder step for autoregressive generation.

        Args:
            token:  Current token index, shape (batch, 1).
            hidden: Current LSTM state (h, c).

        Returns:
            logits: (batch, 1, vocab_size)
            hidden: Updated (h_n, c_n).
        """
        return self.decoder(token, hidden)
