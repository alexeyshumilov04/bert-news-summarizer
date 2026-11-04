# 📰 BERT News Summarizer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Abstractive summarization of Russian news with a pretrained BERT encoder and a
**from-scratch Transformer decoder**, trained end-to-end on
[`IlyaGusev/gazeta`](https://huggingface.co/datasets/IlyaGusev/gazeta). Supports
four decoding strategies — greedy, top-k, nucleus, beam search — all implemented
from scratch. Despite the lack of a pretrained decoder, the model outperforms an
off-the-shelf `ruT5-small` on BLEU and BERTScore.

---

## ✨ Key Features

- **Custom encoder-decoder**: `deepvk/bert-base-uncased` encoder + 3-layer decoder with masked self-attention, cross-attention and FFN, written in PyTorch.
- **Four decoding strategies** as pure functions: `greedy`, `top_k`, `nucleus`, `beam` (length-normalized).
- **Reproducible training**: TensorBoard logging, best-checkpoint saving, per-epoch loss curve.
- **Automatic evaluation**: ROUGE-1/2/L, BLEU, BERTScore on a fixed held-out split.
- **Baseline comparison** against `cointegrated/rut5-small` on the exact same split.
- **Single-config-driven** (`config.yaml`); scripts are thin CLIs over `src/`.

---

## 📂 Project Structure

```text
bert-news-summarizer/
├── config.yaml          # All hyperparameters
├── requirements.txt
├── train.py             # Entry point: training
├── evaluate.py          # Entry point: ROUGE / BLEU / BERTScore
├── generate.py          # Entry point: inference
├── hw_04.ipynb          # Original prototype notebook (kept for reference)
└── src/
    ├── utils.py         # Seed, device, config loading, plotting
    ├── data.py          # gazeta loading, tokenization, DataLoaders
    ├── model.py         # DecoderLayer + BertSummarizer
    ├── generation.py    # greedy / top-k / nucleus / beam
    ├── training.py      # Training loop, TensorBoard, checkpoints
    ├── metrics.py       # ROUGE / BLEU / BERTScore + evaluation
    └── baseline.py      # ruT5-small baseline
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/alexeyshumilov04/News-Summarizer.git
cd bert-news-summarizer
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+, ~2 GB disk. GPU recommended (≥ 8 GB VRAM).
Datasets and weights are downloaded from the HF Hub on first run.

---

## 🛠️ Usage

```bash
# 1. Train (6 epochs AdamW, best-val checkpoint + TensorBoard)
python train.py --config config.yaml
tensorboard --logdir artifacts/runs

# 2. Evaluate — custom model, baseline, or both
python evaluate.py --which custom --strategy top_k
python evaluate.py --which baseline
python evaluate.py --which both --strategy nucleus

# 3. Generate — from text or from a dataset example
python generate.py --text "ЦБ РФ повысил ключевую ставку до 16%..." --strategy nucleus
python generate.py --dataset-index 5 --strategy beam
```

Outputs go to `artifacts/`: `best_model.pt`, `tokenizer/`, `loss_curve.png`, `runs/`.

---

## 🧩 Configuration

All hyperparameters live in `config.yaml`, grouped by concern:

| Section      | Controls                                                |
|--------------|---------------------------------------------------------|
| `seed`       | Global RNG seed                                         |
| `device`     | `auto` / `cuda` / `cpu`                                 |
| `data`       | Dataset, split, max lengths, batch size, test ratio     |
| `model`      | BERT checkpoint, decoder depth, heads, dropout          |
| `baseline`   | Reference seq2seq model                                 |
| `training`   | Epochs, LR, weight decay, output paths                  |
| `generation` | `max_len`, default strategy, top-k / top-p / beams      |

---

## 📊 Results

15 % slice of `IlyaGusev/gazeta` (~7 773 train / ~1 372 test). All rows use the
same held-out split and reference summaries. **Bold** = best per column.

| Model / decoding                      | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU       | BERTScore-F1 |
|---------------------------------------|--------:|--------:|--------:|-----------:|-------------:|
| BERT+decoder · greedy                 | **0.0295** | **0.0083** | **0.0287** | 0.0529 | 0.6669       |
| BERT+decoder · top-k                  | 0.0262  | 0.0054  | 0.0253  | **0.0584** | 0.6991       |
| BERT+decoder · nucleus                | 0.0147  | 0.0015  | 0.0141  | 0.0531     | **0.7003**   |
| `cointegrated/rut5-small` (zero-shot) | 0.0279  | 0.0070  | 0.0277  | 0.0079     | 0.6417       |

**The custom model beats the `ruT5-small` baseline on the metrics that matter
most** — BLEU × 7.4 and BERTScore +0.057 (see the top-k row) — despite `ruT5`
being a purpose-built Russian seq2seq model and despite our model having only
6 epochs on ~7.8 k examples. The advantage comes from domain fine-tuning, not
model size (`BertSummarizer` ≈ 197 M params vs. `ruT5-small` ≈ 64 M). The
slightly lower ROUGE for top-k is expected: the custom model paraphrases,
while `ruT5` copies tokens — which is why greedy decoding keeps the highest
ROUGE.

Among decoding strategies, `top_k` gives the best BLEU / BERTScore balance;
`nucleus` maximizes semantic similarity but loses lexical overlap; `beam`
(length-normalized) is used mainly for qualitative inspection.

---

## 📝 Notes and Limitations

- Reproduction of the architecture, not a SOTA summarizer — with ~7.8 k examples
  it learns topic and style, but summaries are not fully grammatical.
- BERT tokenizer splits Russian words into subwords (`##`-suffixes leak into the
  output). A SentencePiece tokenizer would fit the decoder better.
- Encoder is fine-tuned jointly with the decoder; freezing it is a natural next
  step and would cut training time significantly.
- Decoder starts from `[CLS]`, stops at `[SEP]`; no explicit `[EOS]`.
- No length normalization during training — biases the model towards short
  summaries.

---

## 📄 License

MIT.

