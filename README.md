# Domain-Adaptive Multi-Model Research Paper Summarization System

A domain-aware research paper summarization system that classifies papers as **General** or **Medical** and routes them to the best-performing domain-specific summarization model. Built as a group project for **EC 7203 — Advanced Artificial Intelligence**, University of Ruhuna, Faculty of Engineering.

---

## Research Contributions

- **Angle 1 — Comparative Evaluation Benchmark:** Empirical comparison of pretrained vs. fine-tuned summarization models (BART, PEGASUS) across General and Medical domains using ROUGE and BERTScore metrics.
- **Angle 2 — Domain-Aware Routing System:** A fine-tuned BERT classifier (99.4% accuracy) that automatically routes each paper to its optimal summarization model, validated by the Angle 1 benchmark.

---

## System Architecture

```
Input (PDF / raw text)
        │
        ▼
BERT Domain Classifier (outputs/classifier/)
        │
        ├── General ──▶ BART fine-tuned (outputs/bart_finetuned/)
        └── Medical ──▶ PEGASUS zero-shot (google/pegasus-pubmed)
        │
        ▼
Structured Output Generator (OpenAI GPT-4o-mini)
        │
        ▼
FastAPI Backend + Next.js Frontend
```

---

## Models Used

| Model | Checkpoint | Purpose |
|---|---|---|
| `bert-base-uncased` (fine-tuned) | `outputs/classifier/` | Domain classification |
| `facebook/bart-large-cnn` (fine-tuned) | `outputs/bart_finetuned/` | General paper summarization |
| `google/pegasus-pubmed` (zero-shot) | HuggingFace Hub | Medical paper summarization |

---

## Prerequisites

### System Requirements

- **OS:** Linux (Ubuntu 22.04+ recommended)
- **Python:** 3.11+
- **GPU:** NVIDIA RTX 5090 or any CUDA-capable GPU with 8GB+ VRAM
- **CUDA:** 12.8+ (nightly PyTorch required for Blackwell sm_120 architecture)
- **Node.js:** 22.x+
- **RAM:** 16GB+ recommended

### System-level dependencies

Install required system libraries for WeasyPrint (PDF generation):

```bash
sudo apt install -y \
  libpango-1.0-0 \
  libpangoft2-1.0-0 \
  libpangocairo-1.0-0 \
  libgdk-pixbuf-xlib-2.0-0 \
  libcairo2 \
  libffi-dev
```

Install Node.js 22.x (required for the Next.js frontend):

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

---

## Project Structure

```
research-summarizer/
├── app/
│   ├── backend/
│   │   ├── main.py                  # FastAPI app + lifespan model loading
│   │   ├── dependencies.py          # ResearchSummarizer singleton injection
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── templates/
│   │   │   └── summary.html         # Jinja2 PDF export template
│   │   ├── routes/
│   │   │   └── summarize.py         # API route definitions
│   │   └── controllers/
│   │       ├── summarize.py         # PDF extraction + pipeline logic
│   │       └── pdf.py               # WeasyPrint PDF generation
│   └── frontend/                    # Next.js App Router frontend
│       ├── src/
│       │   ├── app/
│       │   │   ├── layout.tsx
│       │   │   ├── page.tsx
│       │   │   └── globals.css
│       │   ├── components/
│       │   │   ├── UploadForm.tsx
│       │   │   ├── ResultCard.tsx
│       │   │   ├── DomainBadge.tsx
│       │   │   ├── EvalPanel.tsx
│       │   │   ├── DownloadButton.tsx
│       │   │   └── LoadingState.tsx
│       │   └── lib/
│       │       └── api.ts
│       ├── package.json
│       └── next.config.js
├── configs/
│   ├── bart_config.yaml
│   ├── pegasus_config.yaml
│   └── classifier_config.yaml
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_domain_classifier.ipynb
│   ├── 04_bart_finetuning.ipynb
│   ├── 05_pegasus_finetuning.ipynb
│   ├── 06_evaluation_and_comparison.ipynb
│   ├── 07_routing_system.ipynb
│   ├── 08_structured_output_prompting.ipynb
│   └── 09_human_evaluation.ipynb
├── outputs/
│   ├── classifier/                  # Saved BERT classifier checkpoint
│   ├── bart_finetuned/              # Saved BART checkpoint
│   └── pegasus_finetuned/           # Saved PEGASUS checkpoint (not used)
├── results/
│   ├── angle1_benchmark.csv
│   ├── angle1_benchmark.json
│   ├── rouge_scores.json
│   ├── bertscores.json
│   ├── human_eval_scores.json
│   ├── human_eval_per_paper.csv
│   ├── human_eval_summary.csv
│   ├── human_eval_domain_comparison.png
│   ├── human_eval_boxplots.png
│   └── structured/                  # Per-paper JSON outputs
├── slurm/
│   ├── run_bart.py
│   ├── run_pegasus.py
│   └── run_classifier.py
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── router.py
│   ├── summarizer.py
│   ├── output_generator.py
│   ├── evaluation.py
│   └── preprocessing.py
├── pyproject.toml
├── uv.toml
└── .env
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd research-summarizer
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 3. Create virtual environment and install dependencies

```bash
uv sync
```

### 4. Install PyTorch nightly (required for RTX 5090 / Blackwell architecture)

> **Important:** The stable PyTorch release does not support the sm_120 Blackwell architecture. You must install the nightly build. Re-run this after every `uv sync`.

```bash
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --prerelease=allow
```

### 5. Download the spaCy language model

```bash
uv run python -m spacy download en_core_web_sm
```

### 6. Download NLTK data

```bash
uv run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 7. Set up environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...

# HuggingFace cache — use absolute path, no tilde
HF_HOME=/home/<your-username>/.cache/huggingface
```

> **Note:** `HF_HOME` must be an absolute path. Tilde (`~`) is not expanded by `os.getenv` and will cause HuggingFace to attempt Hub downloads instead of loading local checkpoints.

### 8. Install frontend dependencies

```bash
cd app/frontend
npm install
cd ../..
```

---

## Running the Application

Open two terminals from the project root (`research-summarizer/`).

### Terminal 1 — Backend

```bash
uv run uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will take **30–60 seconds** to start while it loads all three models (BERT, BART, PEGASUS) into GPU memory. You will see:

```
Loading ResearchSummarizer... (this may take a minute)
DomainRouter initialising on cuda ...
  Loading classifier from outputs/classifier/ ...
  Loading BART from outputs/bart_finetuned/ ...
  Loading PEGASUS from google/pegasus-pubmed ...
DomainRouter ready.
ResearchSummarizer ready.
```

API docs available at: `http://localhost:8000/docs`

### Terminal 2 — Frontend

```bash
cd app/frontend
npm run dev
```

Frontend available at: `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/summarize/pdf` | Upload a PDF and summarize |
| `POST` | `/api/summarize/text` | Submit raw text and summarize |
| `POST` | `/api/evaluate` | Submit human evaluation scores |
| `POST` | `/api/summary/pdf` | Download summary as PDF |

### Example — summarize raw text

```bash
curl -X POST http://localhost:8000/api/summarize/text \
  -H "Content-Type: application/json" \
  -d '{"title": "Attention Is All You Need", "text": "We propose a new simple network architecture..."}'
```

### Example — upload PDF

```bash
curl -X POST http://localhost:8000/api/summarize/pdf \
  -F "title=Attention Is All You Need" \
  -F "file=@paper.pdf"
```

---

## Running the Notebooks

Start Jupyter from the project root so that `src/` imports resolve correctly:

```bash
uv run jupyter notebook
```

Run notebooks in order:

| Notebook | Purpose |
|---|---|
| `01_data_loading_and_eda.ipynb` | Load datasets, explore distributions |
| `02_preprocessing_pipeline.ipynb` | Text cleaning, section extraction |
| `03_domain_classifier.ipynb` | BERT fine-tuning for domain classification |
| `04_bart_finetuning.ipynb` | BART fine-tuning on arXiv |
| `05_pegasus_finetuning.ipynb` | PEGASUS fine-tuning on PubMed |
| `06_evaluation_and_comparison.ipynb` | Angle 1 — ROUGE and BERTScore benchmark |
| `07_routing_system.ipynb` | Angle 2 — end-to-end routing demo |
| `08_structured_output_prompting.ipynb` | Prompt engineering ablation study |
| `09_human_evaluation.ipynb` | Human evaluation analysis and plots |

---

## Datasets

| Dataset | Source | Domain | Size |
|---|---|---|---|
| `ccdv/arxiv-summarization` | HuggingFace Hub | General (AI/CS/Physics) | ~215,000 pairs |
| `ccdv/pubmed-summarization` | HuggingFace Hub | Medical (Biomedical/Clinical) | ~119,000 pairs |

The datasets are downloaded automatically by the notebooks on first run via the HuggingFace `datasets` library.

---

## Known Issues

### PyTorch nightly must be reinstalled after `uv sync`

`uv sync` may overwrite the nightly PyTorch build with the stable version, which does not support the RTX 5090 (Blackwell sm_120). Always re-run the nightly install command after syncing:

```bash
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --prerelease=allow
```

### PEGASUS tokenizer

Always use `PegasusTokenizer` explicitly — do not use `AutoTokenizer` for PEGASUS. `AutoTokenizer` triggers a tiktoken bug in `transformers==4.46.3`:

```python
from transformers import PegasusTokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-pubmed")
```

### HuggingFace cache path

Use an absolute path for `HF_HOME` in `.env`. Do not use `~`:

```env
# Wrong
HF_HOME=~/.cache/huggingface

# Correct
HF_HOME=/home/dark_web/.cache/huggingface
```

### Jupyter kernel caching

If you edit files in `src/` and the changes are not picked up in a running notebook, restart the kernel — Python caches imported modules for the lifetime of the kernel session.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Package manager | `uv` |
| Deep learning | PyTorch (nightly cu128), HuggingFace Transformers |
| Fine-tuning | HuggingFace `Trainer` API, PEFT |
| NLP preprocessing | spaCy, NLTK, KeyBERT, YAKE |
| PDF parsing | PyMuPDF (`fitz`) |
| Evaluation | `rouge-score`, `bert-score`, `scikit-learn` |
| Experiment tracking | Weights & Biases (`wandb`) |
| Backend | FastAPI, Uvicorn |
| PDF export | Jinja2, WeasyPrint |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS, shadcn/ui |
| Structured output | OpenAI GPT-4o-mini |
| Notebooks | Jupyter |

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for structured output generation (GPT-4o-mini) |
| `HF_HOME` | Recommended | Absolute path to HuggingFace model cache directory |

---

*Domain-Adaptive Multi-Model Research Paper Summarization System — EC 7203 Advanced Artificial Intelligence, University of Ruhuna, Faculty of Engineering*
