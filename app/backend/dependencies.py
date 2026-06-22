from __future__ import annotations
from typing import Annotated
from fastapi import Depends, Request
from src.summarizer import ResearchSummarizer


def get_summarizer(request: Request) -> ResearchSummarizer:
    """Retrieve the ResearchSummarizer instance from app state.
    
    Injected into route handlers via FastAPI's Depends() system.
    The instance is created once during lifespan startup and stored
    on app.state — this function just retrieves it per-request.
    """
    return request.app.state.summarizer


SummarizerDep = Annotated[ResearchSummarizer, Depends(get_summarizer)]