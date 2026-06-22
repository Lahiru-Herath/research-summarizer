from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True,
)


def generate_summary_pdf(result: dict) -> bytes:
    """Render a summary result dict into a PDF using the Jinja2 template.

    Loads summary.html, injects all result fields, renders to a full
    HTML string, then converts to PDF bytes via WeasyPrint.

    Args:
        result: The structured output dict from ResearchSummarizer.
                Must contain: title, domain, confidence, low_confidence,
                probabilities, model_used, summary, key_findings,
                methodology, limitations, future_work.

    Returns:
        PDF as raw bytes, ready to stream to the client.
    """
    template = _jinja_env.get_template("summary.html")

    rendered_html = template.render(
        title=result["title"],
        domain=result["domain"],
        confidence=result["confidence"],
        low_confidence=result["low_confidence"],
        probabilities=result["probabilities"],
        model_used=result["model_used"],
        summary=result["summary"],
        key_findings=result["key_findings"],
        methodology=result["methodology"],
        limitations=result["limitations"],
        future_work=result["future_work"],
        generated_at=datetime.now().strftime("%B %d, %Y at %H:%M"),
    )

    pdf_bytes = HTML(string=rendered_html).write_pdf()
    return pdf_bytes