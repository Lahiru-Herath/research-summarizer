from __future__ import annotations

import json
from pathlib import Path

import fitz  # pymupdf

from src.summarizer import ResearchSummarizer

HUMAN_EVAL_PATH = Path("../../../results/human_evaluations.json")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using PyMuPDF.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        ValueError: If no text could be extracted from the PDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The file may be scanned or image-based."
        )
    return text


def run_summarize(
    text: str,
    title: str,
    summarizer: ResearchSummarizer,
) -> dict:
    """Run the full summarization pipeline on extracted text.

    Args:
        text:       Full article text.
        title:      Paper title string.
        summarizer: Loaded ResearchSummarizer instance from app state.

    Returns:
        Full structured output dict from the pipeline.
    """
    return summarizer.summarize(text=text, title=title)


def save_evaluation(
    title: str,
    domain: str,
    fluency: int,
    relevance: int,
    conciseness: int,
) -> None:
    """Append a human evaluation score entry to the results JSON file.

    Loads existing scores if the file exists, appends the new entry,
    and writes back. Creates the file and parent dirs if needed.

    Args:
        title:       Paper title the evaluation is for.
        domain:      Domain label ("General" or "Medical").
        fluency:     Fluency score (1–5).
        relevance:   Relevance score (1–5).
        conciseness: Conciseness score (1–5).
    """
    HUMAN_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if HUMAN_EVAL_PATH.exists():
        with open(HUMAN_EVAL_PATH, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append({
        "title":       title,
        "domain":      domain,
        "fluency":     fluency,
        "relevance":   relevance,
        "conciseness": conciseness,
    })

    with open(HUMAN_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)