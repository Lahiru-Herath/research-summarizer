"""
output_generator.py
-------------------
Phase 8 — Structured Output Generator.

Takes the raw summary from the DomainRouter and uses OpenAI GPT
to extract structured fields from the paper, producing a rich,
human-readable research digest.

Output schema:
    {
        "title":             str,
        "domain":            str,
        "confidence":        float,
        "model_used":        str,
        "summary":           str,
        "key_findings":      list[str],
        "methodology":       str,
        "limitations":       list[str],
        "future_work":       list[str],
        "low_confidence":    bool,
        "probabilities":     dict,
    }

Usage:
    from src.router import DomainRouter
    from src.output_generator import StructuredOutputGenerator

    router    = DomainRouter()
    generator = StructuredOutputGenerator()

    result = generator.generate(
        text   = article_text,
        title  = "Attention Is All You Need",
        router = router,
    )
    print(result["key_findings"])
    print(result["methodology"])
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

from src.router import DomainRouter


# ── Constants ─────────────────────────────────────────────────────────────────

OPENAI_MODEL    = "gpt-4o-mini"   # fast + cheap, sufficient for structured extraction
MAX_TOKENS      = 1024
RESULTS_DIR     = Path("results/structured")
RETRY_ATTEMPTS  = 3
RETRY_DELAY_SEC = 2

EXTRACTION_PROMPT = """\
You are a scientific paper analyst. Given a paper title, domain, and summary, \
extract structured information and respond ONLY with a valid JSON object — \
no preamble, no markdown fences, no explanation.

Required JSON schema:
{{
  "key_findings":  ["finding 1", "finding 2", ...],
  "methodology":   "...",
  "limitations":   ["limitation 1", ...],
  "future_work":   ["direction 1", ...]
}}

Rules:
- key_findings : 3-5 key contributions or results as short strings
- methodology  : 1-2 sentences describing the methods used
- limitations  : 2-3 limitations of the work as short strings
- future_work  : 2-3 suggested future research directions as short strings

Paper title : {title}
Domain      : {domain}
Summary     : {summary}
"""


class StructuredOutputGenerator:
    """Extracts structured fields from a paper summary using OpenAI GPT.

    Wraps the DomainRouter pipeline and adds LLM-based structured
    extraction on top of the generated summary.

    Args:
        save_outputs: If True, saves each result as a JSON file under
                      results/structured/. Defaults to True.

    Example:
        >>> router    = DomainRouter()
        >>> generator = StructuredOutputGenerator()
        >>> result    = generator.generate(
        ...     text   = full_article_text,
        ...     title  = "BERT: Pre-training of Deep Bidirectional Transformers",
        ...     router = router,
        ... )
        >>> result["key_findings"]
        ['BERT achieves state-of-the-art on 11 NLP tasks', ...]
    """

    def __init__(self, save_outputs: bool = True) -> None:
        self.client       = OpenAI()   # reads OPENAI_API_KEY from env
        self.save_outputs = save_outputs
        if save_outputs:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        text:   str,
        title:  str,
        router: DomainRouter,
    ) -> dict:
        """Full pipeline: route → summarize → extract structured fields.

        Args:
            text:   Full article text. The router classifier uses the
                    first 512 tokens; the summarizer uses up to 1024.
            title:  Paper title string. Used in the extraction prompt
                    and as the output filename if save_outputs is True.
            router: An initialised DomainRouter instance. Pass an
                    existing one to avoid reloading models between calls.

        Returns:
            Dict with keys: title, domain, confidence, model_used,
            summary, key_findings, methodology, limitations,
            future_work, low_confidence, probabilities.
        """
        # Step 1 — route and summarize
        print(f"  Routing: '{title[:60]}...'" if len(title) > 60 else f"  Routing: '{title}'")
        route_result = router.route(text)

        print(f"  Domain     : {route_result['domain']} "
              f"(confidence {route_result['confidence']:.4f})")
        print(f"  Model used : {route_result['model_used']}")
        print(f"  Extracting structured fields via OpenAI ...")

        # Step 2 — extract structured fields with GPT
        structured = self._extract_with_openai(
            title   = title,
            domain  = route_result["domain"],
            summary = route_result["summary"],
        )

        # Step 3 — assemble final output
        output = {
            "title":          title,
            "domain":         route_result["domain"],
            "confidence":     route_result["confidence"],
            "low_confidence": route_result["low_confidence"],
            "probabilities":  route_result["probabilities"],
            "model_used":     route_result["model_used"],
            "summary":        route_result["summary"],
            **structured,
        }

        # Step 4 — optionally save to disk
        if self.save_outputs:
            self._save(title, output)

        return output

    def generate_batch(
        self,
        papers: list[dict],
        router: DomainRouter,
    ) -> list[dict]:
        """Process a batch of papers through the full pipeline.

        Args:
            papers: List of dicts, each with keys "text" and "title".
            router: An initialised DomainRouter instance.

        Returns:
            List of structured output dicts, one per paper.
        """
        results = []
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}]")
            try:
                result = self.generate(
                    text   = paper["text"],
                    title  = paper["title"],
                    router = router,
                )
                results.append(result)
            except Exception as e:
                print(f"  ERROR on paper '{paper['title']}': {e}")
                results.append({"title": paper["title"], "error": str(e)})
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_with_openai(
        self,
        title:   str,
        domain:  str,
        summary: str,
    ) -> dict:
        """Call OpenAI GPT to extract structured fields from a summary.

        Retries up to RETRY_ATTEMPTS times on JSON parse failure.

        Args:
            title:   Paper title.
            domain:  Classified domain ("General" or "Medical").
            summary: Model-generated summary string.

        Returns:
            Dict with keys: key_findings, methodology, limitations,
            future_work.

        Raises:
            RuntimeError: If all retry attempts fail to produce valid JSON.
        """
        prompt = EXTRACTION_PROMPT.format(
            title   = title,
            domain  = domain,
            summary = summary,
        )

        last_error = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                response = self.client.chat.completions.create(
                    model       = OPENAI_MODEL,
                    max_tokens  = MAX_TOKENS,
                    temperature = 0.1,    # low temperature for consistent JSON
                    messages    = [
                        {
                            "role":    "system",
                            "content": "You are a scientific paper analyst. "
                                       "Always respond with valid JSON only.",
                        },
                        {
                            "role":    "user",
                            "content": prompt,
                        },
                    ],
                    response_format = {"type": "json_object"},  # forces JSON mode
                )
                raw  = response.choices[0].message.content.strip()
                data = json.loads(raw)
                return self._validate_structure(data)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = e
                print(f"  Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}. Retrying ...")
                time.sleep(RETRY_DELAY_SEC)

        raise RuntimeError(
            f"Failed to extract structured output after {RETRY_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        )

    def _validate_structure(self, data: dict) -> dict:
        """Validate and normalise the OpenAI JSON response.

        Ensures all required keys are present and have correct types.
        Falls back to safe defaults for any missing or malformed fields.

        Args:
            data: Raw parsed JSON dict from OpenAI.

        Returns:
            Validated dict with keys: key_findings, methodology,
            limitations, future_work.
        """
        def ensure_list(val, default: list) -> list:
            if isinstance(val, list) and len(val) > 0:
                return [str(v) for v in val]
            return default

        def ensure_str(val, default: str) -> str:
            if isinstance(val, str) and val.strip():
                return val.strip()
            return default

        return {
            "key_findings": ensure_list(
                data.get("key_findings"),
                ["No key findings extracted."],
            ),
            "methodology": ensure_str(
                data.get("methodology"),
                "Methodology not extracted.",
            ),
            "limitations": ensure_list(
                data.get("limitations"),
                ["No limitations extracted."],
            ),
            "future_work": ensure_list(
                data.get("future_work"),
                ["No future work extracted."],
            ),
        }

    def _save(self, title: str, output: dict) -> None:
        """Save a structured output dict to a JSON file.

        Filename is derived from the paper title (sanitised).

        Args:
            title:  Paper title used to generate the filename.
            output: Full structured output dict to serialise.
        """
        safe_name = re.sub(r"[^\w\s-]", "", title).strip()
        safe_name = re.sub(r"[\s]+", "_", safe_name)[:80]
        path      = RESULTS_DIR / f"{safe_name}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  Saved → {path}")