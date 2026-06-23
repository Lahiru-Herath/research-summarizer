from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routes.summarize import router as summarize_router
from src.summarizer import ResearchSummarizer

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup and release on shutdown.

    ResearchSummarizer boots DomainRouter (BERT + BART + PEGASUS)
    and StructuredOutputGenerator (OpenAI client). This takes ~30–60s
    on first run as models are moved to GPU.
    """
    # print("Loading ResearchSummarizer... (this may take a minute)")
    app.state.summarizer = ResearchSummarizer(save_outputs=True)
    # print("ResearchSummarizer ready.")
    yield
    # print("Shutting down — releasing model memory.")
    del app.state.summarizer


app = FastAPI(
    title="Research Summarizer API",
    description="Domain-adaptive research paper summarization system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summarize_router, prefix="/api")