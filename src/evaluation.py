"""
evaluation.py
-------------
Angle 1 — Comparative Evaluation Framework.

Evaluates all 6 benchmark rows against their respective test sets:

    Row 1 : BART zero-shot          → arXiv test set   (General)
    Row 2 : BART fine-tuned         → arXiv test set   (General)
    Row 3 : PEGASUS zero-shot       → PubMed test set  (Medical)
    Row 4 : PEGASUS fine-tuned      → PubMed test set  (Medical)
    Row 5 : BART fine-tuned         → PubMed test set  (cross-domain)
    Row 6 : PEGASUS fine-tuned      → arXiv test set   (cross-domain)

Results saved to:
    results/rouge_scores.json
    results/bertscores.json
    results/angle1_benchmark.json   ← full combined table
    results/angle1_benchmark.csv    ← for report copy-paste

Usage:
    uv run python src/evaluation.py
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as bert_score_fn
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PegasusTokenizer,
)

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Model checkpoint paths
BART_FINETUNED    = "outputs/bart_finetuned"
PEGASUS_FINETUNED = "outputs/pegasus_finetuned"
BART_ZEROSHOT     = "facebook/bart-large-cnn"
PEGASUS_ZEROSHOT  = "google/pegasus-pubmed"

# Generation config
BART_MAX_INPUT    = 1024
PEGASUS_MAX_INPUT = 1024
MAX_NEW_TOKENS    = 256
BATCH_SIZE        = 8    # reduce to 4 if you hit OOM during generation

# BERTScore model — use a lightweight model to keep eval fast
BERTSCORE_MODEL   = "distilbert-base-uncased"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, is_pegasus: bool):
    """Load a seq2seq model and its tokenizer from a path or HF hub name.

    Args:
        model_path: Local checkpoint path or HuggingFace model name.
        is_pegasus: If True, loads PegasusTokenizer (SentencePiece-based)
                    to avoid the tiktoken conversion bug in transformers 4.46.

    Returns:
        Tuple of (model, tokenizer), both ready for inference.
    """
    print(f"  Loading tokenizer from {model_path} ...")
    if is_pegasus:
        tokenizer = PegasusTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"  Loading model from {model_path} ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


def generate_summaries(
    model,
    tokenizer,
    articles: list[str],
    max_input: int,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    """Generate summaries for a list of articles in batches.

    Args:
        model:      Loaded seq2seq model on the correct device.
        tokenizer:  Matching tokenizer.
        articles:   List of input article strings.
        max_input:  Maximum input token length (truncates beyond this).
        batch_size: Number of articles to process per forward pass.

    Returns:
        List of decoded summary strings, one per input article.
    """
    summaries = []
    for i in tqdm(range(0, len(articles), batch_size), desc="    Generating", leave=False):
        batch = articles[i : i + batch_size]
        inputs = tokenizer(
            batch,
            max_length=max_input,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        summaries.extend(decoded)

    return summaries


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: List of model-generated summary strings.
        references:  List of ground-truth summary strings.

    Returns:
        Dict with keys rouge1, rouge2, rougeL — each a float (0–1).
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": round(sum(r1) / len(r1), 4),
        "rouge2": round(sum(r2) / len(r2), 4),
        "rougeL": round(sum(rl) / len(rl), 4),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore precision, recall, and F1.

    Args:
        predictions: List of model-generated summary strings.
        references:  List of ground-truth summary strings.

    Returns:
        Dict with keys bertscore_p, bertscore_r, bertscore_f1 — each a float.
    """
    print("    Computing BERTScore (this takes a while) ...")
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=BERTSCORE_MODEL,
        device=DEVICE,
        verbose=False,
    )
    return {
        "bertscore_p":  round(P.mean().item(), 4),
        "bertscore_r":  round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def evaluate_row(
    row_name: str,
    model,
    tokenizer,
    articles: list[str],
    references: list[str],
    max_input: int,
) -> dict:
    """Run full evaluation for one benchmark row.

    Generates summaries, then computes ROUGE and BERTScore.

    Args:
        row_name:   Human-readable label for this benchmark row.
        model:      Loaded seq2seq model.
        tokenizer:  Matching tokenizer.
        articles:   Input articles from the test set.
        references: Ground-truth abstracts from the test set.
        max_input:  Max input token length for this model.

    Returns:
        Dict with row_name, rouge1, rouge2, rougeL, bertscore_f1,
        bertscore_p, bertscore_r.
    """
    print(f"\n{'─' * 60}")
    print(f"  Row : {row_name}")
    print(f"  N   : {len(articles)} samples")
    print(f"{'─' * 60}")

    t0 = time.time()
    predictions = generate_summaries(model, tokenizer, articles, max_input)
    rouge = compute_rouge(predictions, references)
    bscore = compute_bertscore(predictions, references)
    elapsed = time.time() - t0

    result = {"row": row_name, **rouge, **bscore, "elapsed_sec": round(elapsed)}
    print(f"  ROUGE-1: {rouge['rouge1']:.4f}  "
          f"ROUGE-2: {rouge['rouge2']:.4f}  "
          f"ROUGE-L: {rouge['rougeL']:.4f}  "
          f"BERTScore-F1: {bscore['bertscore_f1']:.4f}")
    return result


# ── Load datasets ─────────────────────────────────────────────────────────────

print("\nLoading arXiv test set ...")
arxiv_test  = load_dataset("ccdv/arxiv-summarization", split="test")
arxiv_articles   = arxiv_test["article"]
arxiv_references = arxiv_test["abstract"]

print("Loading PubMed test set ...")
pubmed_test = load_dataset("ccdv/pubmed-summarization", split="test")
pubmed_articles   = pubmed_test["article"]
pubmed_references = pubmed_test["abstract"]

print(f"arXiv  test : {len(arxiv_articles):,} samples")
print(f"PubMed test : {len(pubmed_articles):,} samples")

# ── Run all 6 benchmark rows ──────────────────────────────────────────────────

results = []

# Row 1 — BART zero-shot on arXiv (General, no fine-tuning)
print("\n[1/6] BART zero-shot → arXiv")
model, tokenizer = load_model_and_tokenizer(BART_ZEROSHOT, is_pegasus=False)
results.append(evaluate_row(
    "BART zero-shot | General (arXiv)",
    model, tokenizer, arxiv_articles, arxiv_references, BART_MAX_INPUT,
))
del model; torch.cuda.empty_cache()

# Row 2 — BART fine-tuned on arXiv (General, in-domain)
print("\n[2/6] BART fine-tuned → arXiv")
model, tokenizer = load_model_and_tokenizer(BART_FINETUNED, is_pegasus=False)
results.append(evaluate_row(
    "BART fine-tuned | General (arXiv)",
    model, tokenizer, arxiv_articles, arxiv_references, BART_MAX_INPUT,
))
del model; torch.cuda.empty_cache()

# Row 3 — PEGASUS zero-shot on PubMed (Medical, no fine-tuning)
print("\n[3/6] PEGASUS zero-shot → PubMed")
model, tokenizer = load_model_and_tokenizer(PEGASUS_ZEROSHOT, is_pegasus=True)
results.append(evaluate_row(
    "PEGASUS zero-shot | Medical (PubMed)",
    model, tokenizer, pubmed_articles, pubmed_references, PEGASUS_MAX_INPUT,
))
del model; torch.cuda.empty_cache()

# Row 4 — PEGASUS fine-tuned on PubMed (Medical, in-domain)
print("\n[4/6] PEGASUS fine-tuned → PubMed")
model, tokenizer = load_model_and_tokenizer(PEGASUS_FINETUNED, is_pegasus=True)
results.append(evaluate_row(
    "PEGASUS fine-tuned | Medical (PubMed)",
    model, tokenizer, pubmed_articles, pubmed_references, PEGASUS_MAX_INPUT,
))
del model; torch.cuda.empty_cache()

# Row 5 — BART fine-tuned on PubMed (cross-domain: General model on Medical data)
print("\n[5/6] BART fine-tuned → PubMed (cross-domain)")
model, tokenizer = load_model_and_tokenizer(BART_FINETUNED, is_pegasus=False)
results.append(evaluate_row(
    "BART fine-tuned | Medical (cross-domain)",
    model, tokenizer, pubmed_articles, pubmed_references, BART_MAX_INPUT,
))
del model; torch.cuda.empty_cache()

# Row 6 — PEGASUS fine-tuned on arXiv (cross-domain: Medical model on General data)
print("\n[6/6] PEGASUS fine-tuned → arXiv (cross-domain)")
model, tokenizer = load_model_and_tokenizer(PEGASUS_FINETUNED, is_pegasus=True)
results.append(evaluate_row(
    "PEGASUS fine-tuned | General (cross-domain)",
    model, tokenizer, arxiv_articles, arxiv_references, PEGASUS_MAX_INPUT,
))
del model; torch.cuda.empty_cache()

# ── Save results ──────────────────────────────────────────────────────────────

print("\n\nSaving results ...")

# Separate ROUGE and BERTScore files (matches architecture doc)
rouge_out = [{k: v for k, v in r.items()
              if k in ("row", "rouge1", "rouge2", "rougeL")} for r in results]
bert_out  = [{k: v for k, v in r.items()
              if k in ("row", "bertscore_p", "bertscore_r", "bertscore_f1")} for r in results]

with open(RESULTS_DIR / "rouge_scores.json", "w") as f:
    json.dump(rouge_out, f, indent=2)

with open(RESULTS_DIR / "bertscores.json", "w") as f:
    json.dump(bert_out, f, indent=2)

with open(RESULTS_DIR / "angle1_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)

# CSV for easy copy-paste into report
df = pd.DataFrame(results).drop(columns=["elapsed_sec"])
df.to_csv(RESULTS_DIR / "angle1_benchmark.csv", index=False)

print("\n✓ Results saved:")
print("  results/rouge_scores.json")
print("  results/bertscores.json")
print("  results/angle1_benchmark.json")
print("  results/angle1_benchmark.csv")

# ── Print final table ─────────────────────────────────────────────────────────

print("\n\n" + "═" * 90)
print("ANGLE 1 — BENCHMARK RESULTS")
print("═" * 90)
print(df.to_string(index=False))
print("═" * 90)