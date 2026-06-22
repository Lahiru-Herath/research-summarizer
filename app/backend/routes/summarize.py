from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from typing import Optional

from app.backend.controllers.summarize import (
    extract_text_from_pdf,
    run_summarize,
    save_evaluation,
)
from app.backend.dependencies import SummarizerDep
from app.backend.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    SummarizeResponse,
    SummarizeTextRequest,
)
from app.backend.controllers.pdf import generate_summary_pdf

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/summarize/pdf", response_model=SummarizeResponse)
async def summarize_pdf(
    summarizer: SummarizerDep,
    file: UploadFile = File(...),
    title: str = Form(...),
) -> SummarizeResponse:
    """Accept a PDF upload and run the full summarization pipeline.

    Extracts text from the PDF, routes through the domain classifier,
    generates a summary, and extracts structured fields via GPT.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result = run_summarize(text=text, title=title, summarizer=summarizer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    return SummarizeResponse(**result)


@router.post("/summarize/text", response_model=SummarizeResponse)
def summarize_text(
    body: SummarizeTextRequest,
    summarizer: SummarizerDep,
) -> SummarizeResponse:
    """Accept raw text and run the full summarization pipeline.

    Expects a JSON body with 'title' and 'text' fields.
    """
    try:
        result = run_summarize(
            text=body.text,
            title=body.title,
            summarizer=summarizer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    return SummarizeResponse(**result)


@router.post("/evaluate", response_model=EvaluationResponse)
def submit_evaluation(body: EvaluationRequest) -> EvaluationResponse:
    """Save a human evaluation score entry to results/human_eval_scores.json."""
    for field, val in [
        ("fluency", body.fluency),
        ("relevance", body.relevance),
        ("conciseness", body.conciseness),
    ]:
        if not (1 <= val <= 5):
            raise HTTPException(
                status_code=422,
                detail=f"'{field}' must be between 1 and 5.",
            )

    save_evaluation(
        title=body.title,
        domain=body.domain,
        fluency=body.fluency,
        relevance=body.relevance,
        conciseness=body.conciseness,
    )
    return EvaluationResponse(saved=True)


@router.post("/summary/pdf")
def download_summary_pdf(body: SummarizeResponse) -> Response:
    """Accept a summary result and return it as a downloadable PDF."""
    try:
        pdf_bytes = generate_summary_pdf(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    # Strip any non-ASCII characters from the filename before putting
    # it in the Content-Disposition header — HTTP headers are latin-1
    # only and special characters like en-dashes will crash encoding.
    safe_title = (
        body.title[:60]
        .encode("ascii", errors="ignore")
        .decode("ascii")
        .replace(" ", "_")
        .replace("/", "-")
        .strip("_")
    ) or "summary"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_title}_summary.pdf"'
        },
    )