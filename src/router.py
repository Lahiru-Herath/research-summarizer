"""
router.py
---------
Domain-Aware Model Router — Angle 2 of the research system.

Routes each input paper to its optimal summarization model based on
domain classification by the fine-tuned BERT classifier:

    Input text
        │
        ▼
    BERT Classifier (outputs/classifier/)
        │
        ├── General (confidence ≥ threshold) ──▶ BART fine-tuned (arXiv)
        └── Medical (confidence ≥ threshold) ──▶ PEGASUS zero-shot (PubMed)

Usage:
    from src.router import DomainRouter

    router = DomainRouter()
    result = router.route("Transformer-based attention mechanisms for...")
    print(result["summary"])
    print(result["domain"])
    print(result["confidence"])
    print(result["model_used"])
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PegasusTokenizer,
)

from src.classifier import load_classifier, predict_domain as clf_predict_domain

# ── Project root resolution ───────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if not (_PROJECT_ROOT / "outputs").exists():
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / "outputs" / "classifier").exists():
            _PROJECT_ROOT = parent
            break

# print(f"[router.py] Project root resolved to: {_PROJECT_ROOT}")

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSIFIER_PATH = str(_PROJECT_ROOT / "outputs" / "classifier")
BART_PATH       = str(_PROJECT_ROOT / "outputs" / "bart_finetuned")
PEGASUS_PATH    = "google/pegasus-pubmed"   # zero-shot — HF Hub

# print(f"Paths:\n  Classifier : {CLASSIFIER_PATH}\n  BART       : {BART_PATH}\n  PEGASUS    : {PEGASUS_PATH}\n")

LABEL_MAP = {0: "General", 1: "Medical"}
SUMMARIZER_MAP = {
    "General": "BART fine-tuned (facebook/bart-large-cnn → arXiv)",
    "Medical": "PEGASUS zero-shot (google/pegasus-pubmed)",
}

CLASSIFIER_MAX_LEN = 512
BART_MAX_INPUT     = 1024
PEGASUS_MAX_INPUT  = 1024
MAX_NEW_TOKENS     = 256
DEFAULT_THRESHOLD  = 0.5


class DomainRouter:
    """Routes a research paper to the best domain-specific summarizer.

    Loads three models at construction time and keeps them in memory:
        - BERT classifier   (src/classifier.py — domain detection)
        - BART fine-tuned   (General / arXiv summarization)
        - PEGASUS zero-shot (Medical / PubMed summarization)

    Args:
        device:    Torch device string. Defaults to "cuda" if available.
        threshold: Minimum classifier softmax confidence to trust the
                   predicted label. Below this the prediction is still
                   used but ``low_confidence`` is flagged in the output.

    Example:
        >>> router = DomainRouter()
        >>> result = router.route("Deep learning for protein folding...")
        >>> result["domain"]
        'Medical'
        >>> result["confidence"]
        0.9823
    """

    def __init__(
        self,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        # print(f"DomainRouter initialising on {self.device} ...")

        # ── Classifier (delegated to src/classifier.py) ───────────────────────
        # print(f"  Loading classifier from {CLASSIFIER_PATH} ...")
        self.clf_pipeline = load_classifier(
            CLASSIFIER_PATH,
            device=str(self.device),
        )

        # ── BART (General) ────────────────────────────────────────────────────
        # print(f"  Loading BART from {BART_PATH} ...")
        self.bart_tokenizer = AutoTokenizer.from_pretrained(BART_PATH)
        self.bart_model     = AutoModelForSeq2SeqLM.from_pretrained(
            BART_PATH
        ).to(self.device).eval()

        # ── PEGASUS (Medical) ─────────────────────────────────────────────────
        # print(f"  Loading PEGASUS from {PEGASUS_PATH} ...")
        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(PEGASUS_PATH)
        self.pegasus_model     = AutoModelForSeq2SeqLM.from_pretrained(
            PEGASUS_PATH
        ).to(self.device).eval()

        # print("DomainRouter ready.\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_domain(self, text: str) -> dict:
        """Classify the domain of an input text.

        Delegates to src/classifier.py and enriches the result with
        full probability distribution and low_confidence flag.

        Args:
            text: Raw input string (title + abstract recommended).

        Returns:
            Dict with keys:
                - ``label``          (str)   : "General" or "Medical"
                - ``label_id``       (int)   : 0 or 1
                - ``confidence``     (float) : softmax probability of predicted class
                - ``low_confidence`` (bool)  : True if confidence < threshold
                - ``probabilities``  (dict)  : {"General": float, "Medical": float}
        """
        # Get label + confidence from classifier.py
        label, confidence = clf_predict_domain(text, self.clf_pipeline)
        label_id = next(k for k, v in LABEL_MAP.items() if v == label)

        # Get full probability distribution for both classes
        encoded = self.clf_pipeline.tokenizer(
            text,
            max_length=CLASSIFIER_MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.clf_pipeline.model(**encoded).logits

        probs = F.softmax(logits, dim=-1).squeeze()

        return {
            "label":          label,
            "label_id":       label_id,
            "confidence":     round(confidence, 4),
            "low_confidence": confidence < self.threshold,
            "probabilities":  {
                "General": round(float(probs[0].item()), 4),
                "Medical":  round(float(probs[1].item()), 4),
            },
        }

    def summarize(self, text: str, domain: str) -> str:
        """Generate a summary using the domain-appropriate model.

        Args:
            text:   Input article or abstract text.
            domain: "General" routes to BART; "Medical" routes to PEGASUS.

        Returns:
            Decoded summary string.

        Raises:
            ValueError: If domain is not "General" or "Medical".
        """
        if domain == "General":
            return self._summarize_bart(text)
        elif domain == "Medical":
            return self._summarize_pegasus(text)
        else:
            raise ValueError(
                f"Unknown domain: '{domain}'. Expected 'General' or 'Medical'."
            )

    def route(self, text: str) -> dict:
        """Full pipeline: classify domain then generate summary.

        This is the main entry point for end-to-end inference.

        Args:
            text: Raw input string. For best classifier accuracy, pass
                  ``title + " [SEP] " + abstract``. For summarization,
                  the full article body is used if available — pass the
                  full text and the classifier will use the first 512
                  tokens while the summarizer uses up to its max input.

        Returns:
            Dict with keys:
                - ``summary``        (str)   : generated summary
                - ``domain``         (str)   : "General" or "Medical"
                - ``confidence``     (float) : classifier confidence (0–1)
                - ``low_confidence`` (bool)  : True if confidence < threshold
                - ``probabilities``  (dict)  : per-class softmax scores
                - ``model_used``     (str)   : human-readable model description
        """
        clf_result = self.predict_domain(text)
        domain     = clf_result["label"]
        summary    = self.summarize(text, domain)

        return {
            "summary":        summary,
            "domain":         domain,
            "confidence":     clf_result["confidence"],
            "low_confidence": clf_result["low_confidence"],
            "probabilities":  clf_result["probabilities"],
            "model_used":     SUMMARIZER_MAP[domain],
        }

    def route_batch(self, texts: list[str]) -> list[dict]:
        """Route a list of texts through the full pipeline.

        Args:
            texts: List of input strings.

        Returns:
            List of result dicts, one per input (same structure as ``route()``).
        """
        return [self.route(text) for text in texts]

    # ── Private summarizers ───────────────────────────────────────────────────

    def _summarize_bart(self, text: str) -> str:
        """Generate a summary using the fine-tuned BART model.

        Args:
            text: Input article text.

        Returns:
            Decoded summary string.
        """
        inputs = self.bart_tokenizer(
            text,
            max_length=BART_MAX_INPUT,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.bart_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.bart_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _summarize_pegasus(self, text: str) -> str:
        """Generate a summary using the PEGASUS zero-shot model.

        Args:
            text: Input article text.

        Returns:
            Decoded summary string.
        """
        inputs = self.pegasus_tokenizer(
            text,
            max_length=PEGASUS_MAX_INPUT,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.pegasus_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.pegasus_tokenizer.decode(output_ids[0], skip_special_tokens=True)