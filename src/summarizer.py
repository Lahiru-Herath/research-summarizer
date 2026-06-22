"""
summarizer.py
-------------
Top-level pipeline entry point for the Domain-Adaptive
Multi-Model Research Paper Summarization System.

Wires together:
    DomainRouter         (src/router.py)
    StructuredOutputGenerator (src/output_generator.py)

into a single callable interface for human evaluation (Phase 9)
and any downstream demo or API layer.

Usage:
    from src.summarizer import ResearchSummarizer

    summarizer = ResearchSummarizer()

    # Single paper
    result = summarizer.summarize(
        text  = "Full article text here...",
        title = "Paper Title",
    )
    print(result["summary"])
    print(result["key_findings"])

    # Batch
    results = summarizer.summarize_batch([
        {"text": "...", "title": "Paper A"},
        {"text": "...", "title": "Paper B"},
    ])
"""

from __future__ import annotations

from src.router import DomainRouter
from src.output_generator import StructuredOutputGenerator


class ResearchSummarizer:
    """End-to-end research paper summarization pipeline.

    Combines domain classification, model routing, summarization,
    and structured field extraction into a single interface.

    Args:
        save_outputs: If True, saves each result as a JSON file under
                      results/structured/. Defaults to True.

    Example:
        >>> s = ResearchSummarizer()
        >>> result = s.summarize(text="...", title="My Paper")
        >>> result.keys()
        dict_keys(['title', 'domain', 'confidence', 'low_confidence',
                   'probabilities', 'model_used', 'summary',
                   'key_findings', 'methodology', 'limitations',
                   'future_work'])
    """

    def __init__(self, save_outputs: bool = True) -> None:
        self.router    = DomainRouter()
        self.generator = StructuredOutputGenerator(save_outputs=save_outputs)

    def summarize(self, text: str, title: str) -> dict:
        """Run the full pipeline on a single paper.

        Args:
            text:  Full article text. Title + abstract is sufficient
                   for classification; more text improves summarization.
            title: Paper title. Used in structured extraction prompt
                   and as the saved JSON filename.

        Returns:
            Dict with keys:
                title, domain, confidence, low_confidence,
                probabilities, model_used, summary,
                key_findings, methodology, limitations, future_work.
        """
        return self.generator.generate(
            text   = text,
            title  = title,
            router = self.router,
        )

    def summarize_batch(self, papers: list[dict]) -> list[dict]:
        """Run the full pipeline on a list of papers.

        Args:
            papers: List of dicts, each with keys "text" and "title".

        Returns:
            List of result dicts, one per paper. Failed papers include
            an "error" key instead of the normal output fields.
        """
        return self.generator.generate_batch(
            papers = papers,
            router = self.router,
        )