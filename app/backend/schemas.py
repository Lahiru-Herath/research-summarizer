from pydantic import BaseModel
from typing import Optional


class SummarizeTextRequest(BaseModel):
    title: str
    text: str


class EvaluationRequest(BaseModel):
    title: str
    domain: str
    fluency: int
    relevance: int
    conciseness: int


class SummarizeResponse(BaseModel):
    title: str
    domain: str
    confidence: float
    low_confidence: bool
    probabilities: dict
    model_used: str
    summary: str
    key_findings: list[str]
    methodology: str
    limitations: list[str]
    future_work: list[str]


class EvaluationResponse(BaseModel):
    saved: bool


class HealthResponse(BaseModel):
    status: str