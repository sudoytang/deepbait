# DeepBait — 5-Minute Presentation Script

**Total time: ~5 minutes**

---

## Title Slide (~15 seconds)

> Good afternoon everyone, today we'll present DeepBait — our project on generating clickbait headlines from article content using deep learning.

---

## Slide 1: Problem & Motivation (~1 minute)

> Clickbait headlines are everywhere on the internet — they're designed to grab attention and drive clicks, often by being sensational or deliberately vague. Most existing research on clickbait focuses on *detection* — essentially a classification task where you label a headline as clickbait or not.
>
> Our project tackles the *opposite* problem: can we *generate* clickbait headlines? And more specifically, can we generate them *conditioned on the actual article content*? This is a much harder task because the model needs to both understand the article and produce a headline in a specific style.
>
> Why does this matter? First, understanding how clickbait can be generated from article content gives us insight into how it exploits readers. Second, generated clickbait can serve as adversarial samples to improve clickbait detectors. And third, it's an interesting testbed for conditional text generation when you have limited labelled data.

---

## Slide 2: Model Architecture (~1 minute 30 seconds)

> For model architecture, we use a Seq2Seq encoder-decoder model built with LSTMs.
>
> The article text goes through tokenization, then an embedding layer, and into an LSTM Encoder. The encoder reads the article — truncated to the first 100 tokens — and compresses it into a hidden state vector, the h_n and c_n pair.
>
> This hidden state is then passed as the initial state to the LSTM Decoder, which generates the clickbait title one word at a time. During training, we use teacher forcing — the decoder receives the ground-truth previous word at each step rather than its own prediction. At inference time, we use autoregressive generation with temperature sampling, where the temperature parameter controls the trade-off between creativity and coherence.
>
> The embedding dimension is 128, hidden dimension is 256, and we use 2 LSTM layers with dropout of 0.3. Encoder and decoder share the same vocabulary so hidden states are directly compatible.
>
> In addition to our custom LSTM model, we also set up a BART baseline — Experiment 3 — where we fine-tune facebook's bart-large-cnn model. Since BART is already pre-trained on CNN/DailyMail summarisation, it already knows how to read an article and produce a short summary. We just need to shift its style toward clickbait.

---

## Slide 3: Experiments & Results (~1 minute 30 seconds)

> Now let's look at our experiments and results.
>
> For data, we combine the Kaggle Clickbait dataset — which gives us about 4,300 article-title pairs — with the Webis-Clickbait-17 corpus, adding another 38,500 pairs. In total, we have about 42,800 training samples.
>
> We ran three experiments. Experiment 1 is direct training — we train our Seq2Seq LSTM from scratch on the combined dataset. This achieves a best validation loss of 6.278, which corresponds to a perplexity of 533. The model converges but the perplexity is high, indicating it struggles to model the full distribution.
>
> In Experiment 2, we take a two-stage approach. First, we pretrain the same LSTM architecture on CNN/DailyMail — a general news summarisation dataset — to teach the model how to read articles and produce short text. Then we fine-tune on our clickbait data. The pretraining stage alone reaches a validation perplexity of 191, and after fine-tuning on clickbait, we get down to a perplexity of 184.
>
> The key finding here is striking: two-stage training reduces perplexity by 65% compared to direct training — from 533 down to 184. This confirms that general summarisation knowledge transfers effectively to clickbait generation, and that the two-stage approach is critical when labelled clickbait data is scarce.
>
> In Experiment 3, we fine-tune BART-large-CNN — a 406 million parameter transformer that's already pre-trained on summarisation. With just one epoch of fine-tuning on our clickbait data, BART achieves a best validation loss of 2.038, corresponding to a perplexity of just 7.7. That's a massive leap — two orders of magnitude better than the direct LSTM, and over 20 times better than the pretrained LSTM. This really demonstrates the power of large-scale pre-trained transformers even on a niche task like clickbait generation.

---

## Slide 4: Conclusion & Future Work (~45 seconds)

> To wrap up, we have three main takeaways. First, article-conditioned generation is far more meaningful than simply generating headlines from seed phrases — the model actually reads and responds to article content. Second, data augmentation through the Webis-17 corpus and two-stage training are critical strategies when working with limited labelled data. And third, BART fine-tuning achieves a perplexity of 7.7, vastly outperforming our best LSTM at 184 — showing that large pre-trained transformers transfer remarkably well even to niche generation tasks.
>
> For future work, we plan to add an attention mechanism to the LSTM so the decoder can focus on the most relevant parts of the article. We also want to replace random sampling with beam search for more consistent generation quality. And finally, we want to conduct human evaluation comparing BART and LSTM outputs, and measure quality with metrics like ROUGE and BERTScore.
>
> Thank you — we're happy to take any questions.
