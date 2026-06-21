"""
classifier.py
-------------
Reusable BERT-based domain classifier for the Domain-Adaptive Research Paper
Summarization System.

Provides:
    - ClassifierPipeline: wraps a fine-tuned bert-base-uncased model
    - load_classifier(checkpoint_dir)  → ClassifierPipeline
    - predict_domain(text, pipeline)   → (label: str, confidence: float)
    - batch_predict(texts, pipeline)   → list[dict]

Label map (mirrors configs/classifier_config.yaml):
    0 → "General"
    1 → "Medical"

Usage
-----
    from src.classifier import load_classifier, predict_domain

    pipeline = load_classifier("outputs/classifier")
    label, conf = predict_domain("Deep learning for protein folding ...", pipeline)
    print(label, conf)  # Medical  0.9732
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────────

LABEL_MAP: dict[int, str] = {0: "General", 1: "Medical"}
DEFAULT_MAX_LENGTH: int = 512


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class ClassifierPipeline:
    """Container for the tokenizer, model, device, and label map.

    Args:
        tokenizer: HuggingFace fast tokenizer (bert-base-uncased).
        model:     Fine-tuned BertForSequenceClassification.
        device:    Torch device (cuda or cpu).
        label_map: Mapping from integer class id to domain string.
        max_length: Maximum tokenizer input length (default 512).
    """

    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device
    label_map: dict[int, str]
    max_length: int = DEFAULT_MAX_LENGTH


# ── Loader ───────────────────────────────────────────────────────────────────


def load_classifier(
    checkpoint_dir: str,
    label_map: Optional[dict[int, str]] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: Optional[str] = None,
) -> ClassifierPipeline:
    """Load a fine-tuned classifier from a checkpoint directory.

    Args:
        checkpoint_dir: Path to the saved HuggingFace checkpoint
                        (e.g. ``"outputs/classifier"``).
        label_map:      Override the default {0: "General", 1: "Medical"} map.
        max_length:     Maximum number of tokens passed to the tokenizer.
        device:         ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).

    Returns:
        ClassifierPipeline ready for inference.
    """
    if label_map is None:
        label_map = LABEL_MAP

    resolved_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(resolved_device)
    model.eval()

    return ClassifierPipeline(
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        label_map=label_map,
        max_length=max_length,
    )


# ── Inference helpers ─────────────────────────────────────────────────────────


def _build_input_text(text: str) -> str:
    """Normalise raw text for the classifier.

    Strips leading/trailing whitespace and collapses inner whitespace runs to
    a single space.  The caller is responsible for prepending a title and
    [SEP] token if desired.

    Args:
        text: Raw paper text (title + abstract, or just abstract).

    Returns:
        Cleaned string ready for tokenization.
    """
    import re

    return re.sub(r"\s+", " ", text).strip()


def predict_domain(
    text: str,
    pipeline: ClassifierPipeline,
) -> tuple[str, float]:
    """Predict the domain of a single paper.

    Args:
        text:     Input string — typically ``title + " [SEP] " + abstract``.
        pipeline: A loaded ClassifierPipeline.

    Returns:
        Tuple of (label, confidence) where label is ``"General"`` or
        ``"Medical"`` and confidence is a float in [0, 1].
    """
    clean = _build_input_text(text)

    encoded = pipeline.tokenizer(
        clean,
        max_length=pipeline.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {k: v.to(pipeline.device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = pipeline.model(**encoded).logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    class_id = int(torch.argmax(probs).item())
    confidence = float(probs[class_id].item())
    label = pipeline.label_map[class_id]

    return label, confidence


def batch_predict(
    texts: list[str],
    pipeline: ClassifierPipeline,
    batch_size: int = 32,
) -> list[dict]:
    """Predict domains for a list of papers in batches.

    Args:
        texts:      List of input strings (title + [SEP] + abstract).
        pipeline:   A loaded ClassifierPipeline.
        batch_size: Number of examples per forward pass (default 32).

    Returns:
        List of dicts, each containing:
            - ``"label"``      (str)   — ``"General"`` or ``"Medical"``
            - ``"confidence"`` (float) — model confidence for the predicted class
            - ``"class_id"``   (int)   — raw integer class index
    """
    results: list[dict] = []
    clean_texts = [_build_input_text(t) for t in texts]

    for start in range(0, len(clean_texts), batch_size):
        batch = clean_texts[start : start + batch_size]

        encoded = pipeline.tokenizer(
            batch,
            max_length=pipeline.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(pipeline.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = pipeline.model(**encoded).logits

        probs = torch.softmax(logits, dim=-1)
        class_ids = torch.argmax(probs, dim=-1).tolist()
        confidences = probs[range(len(batch)), class_ids].tolist()

        for cid, conf in zip(class_ids, confidences):
            results.append(
                {
                    "label": pipeline.label_map[cid],
                    "confidence": round(conf, 4),
                    "class_id": cid,
                }
            )

    return results