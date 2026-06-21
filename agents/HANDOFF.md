# HANDOFF.md — Domain-Adaptive Research Paper Summarization System

## Session Handoff Document — Continue Development in New Chat

---

## How to Use This Document

Paste this entire file into a new Claude chat with the following message:

> "I am continuing development of my AI research project. This HANDOFF.md
> contains the full project state, all decisions made, all code written,
> and exactly what to do next. Please read it carefully and continue from
> Phase 3."

---

## Project Summary

**Domain-Adaptive Multi-Model Research Paper Summarization System**
Built for an Advanced Artificial Intelligence university course (EC 7203).
Running on a Windows laptop for development, moving to HPC for full training.

Two core research contributions:

- **Angle 1 — Comparative Evaluation Framework:** Rigorous empirical benchmark
  comparing pretrained vs fine-tuned summarization models across general and
  medical domains using ROUGE-1, ROUGE-2, ROUGE-L, BERTScore. Cross-domain
  rows are the empirical proof that routing is necessary.
- **Angle 2 — Domain-Aware Model Routing:** Fine-tuned BERT classifier that
  routes each paper to its best domain-specific summarization model.

---

## Environment

| Property        | Value                                       |
| --------------- | ------------------------------------------- |
| OS (dev)        | Windows 11, MINGW64 / Git Bash              |
| Python          | 3.11 (pinned via uv)                        |
| Package manager | `uv` exclusively — no pip commands anywhere |
| Dependency file | `pyproject.toml` (not requirements.txt)     |
| IDE             | VS Code + Jupyter notebooks                 |
| HPC scheduler   | SLURM (moving to HPC today)                 |
| Seed            | 42 everywhere                               |

---

## Current Dataset Mode — IMPORTANT

### Dev Mode (laptop — current state)

The project currently runs in **`DATASET_MODE = "dev"`** using:

- Dataset: `Devika03/Research_Paper_Summarization_Dataset`
- Size: 334 rows, ~1.18 MB
- Only a `train` split exists — manually split 80/10/10 in the loader
- Fields used: `Abstract` + `Conclusion` → `article`, `Summarization` → `summary`, `Topic` → `domain`
- Domain map: `"Medical Data Analysis"` → `"Medical"`, everything else → `"General"`
- Processed splits saved to: `data/processed_dev/train.json`, `validation.json`, `test.json`

### Full Mode (HPC — switch tomorrow)

When moving to HPC, change **one line** in each notebook:

```python
DATASET_MODE = "full"   # was "dev"
```

Full mode uses:

- `ccdv/arxiv-summarization` (~215K pairs) → General domain → fine-tune LED
- `ccdv/pubmed` (~119K pairs) → Medical domain → fine-tune PEGASUS
- Pre-defined train/val/test splits — do NOT re-split

### HPC Migration Checklist

When you get to HPC tomorrow:

- [ ] Clone the repo or `rsync` the project folder to HPC
- [ ] Run `uv venv --python 3.11` and `uv sync` to recreate the environment
- [ ] Set `HF_DATASETS_CACHE` in `.env` to an HPC scratch directory
- [ ] Run `01_data_loading_and_eda.ipynb` with `DATASET_MODE = "full"` first
      — this downloads and saves `data/processed_full/` splits
- [ ] All subsequent notebooks auto-load from `data/processed_full/`
- [ ] Submit SLURM jobs: `train_led_arxiv.sbatch` and `train_pegasus_pubmed.sbatch`
      simultaneously (longest jobs, ~6–8 hrs each)
- [ ] Fill in cluster-specific `module load` commands in all `.sbatch` files
      (marked with `# CLUSTER-SPECIFIC` comments)
- [ ] Set real `WANDB_API_KEY` in `.env`

---

## Decided Architecture — Do Not Revisit

| Decision             | Choice                                          | Reason                                                                |
| -------------------- | ----------------------------------------------- | --------------------------------------------------------------------- |
| General domain model | `allenai/led-large-16384`                       | Handles 16,384 tokens; BART truncates at 1,024 losing most of a paper |
| Medical domain model | `google/pegasus-pubmed`                         | Domain-pretrained on PubMed abstracts                                 |
| Domain classifier    | `bert-base-uncased`                             | Fine-tuned binary classifier (General/Medical)                        |
| Package manager      | `uv` only                                       | No pip commands anywhere in the project                               |
| Dataset (full)       | `ccdv/arxiv-summarization` + `ccdv/pubmed`      | Large-scale, pre-split, well-established benchmarks                   |
| Dataset (dev)        | `Devika03/Research_Paper_Summarization_Dataset` | 1.18 MB, no download cost, same pipeline                              |
| Training framework   | HuggingFace `Trainer` API + `accelerate`        | Standard, wandb integration built-in                                  |
| Angles in scope      | Angle 1 + Angle 2 only                          | Angle 3 (multi-doc) and Angle 4 (richer output) are future work       |

---

## Project Folder Structure

```
research-summarizer/
├── pyproject.toml
├── README.md
├── HANDOFF.md                        ← this file
├── .gitignore
├── .env                              ← not committed; contains HF_DATASETS_CACHE, WANDB_API_KEY
├── data/
│   ├── processed_dev/                ← ✅ exists (Phase 1 output)
│   │   ├── train.json                (264 rows)
│   │   ├── validation.json           (33 rows)
│   │   └── test.json                 (33 rows)
│   └── processed_full/               ← created on HPC after switching to full mode
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb         ✅ COMPLETE
│   ├── 02_preprocessing_pipeline.ipynb       ✅ COMPLETE
│   ├── 03_domain_classifier.ipynb            ← NEXT
│   ├── 04_led_finetuning.ipynb               ← TODO
│   ├── 05_pegasus_finetuning.ipynb           ← TODO
│   ├── 06_evaluation_and_comparison.ipynb    ← TODO
│   ├── 07_routing_system.ipynb               ← TODO
│   ├── 08_structured_output_prompting.ipynb  ← TODO
│   └── 09_human_evaluation.ipynb             ← TODO
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              ✅ COMPLETE (see full code below)
│   ├── classifier.py                 ← TODO (Phase 3)
│   ├── summarizer.py                 ← TODO (Phase 4/5)
│   ├── router.py                     ← TODO (Phase 7)
│   ├── output_generator.py           ← TODO (Phase 8)
│   └── evaluation.py                 ← TODO (Phase 6)
├── slurm/
│   ├── train_led_arxiv.sbatch        ← TODO (Phase 4)
│   ├── train_pegasus_pubmed.sbatch   ← TODO (Phase 5)
│   ├── train_classifier.sbatch       ← TODO (Phase 3)
│   └── evaluate_all.sbatch           ← TODO (Phase 6)
├── configs/
│   ├── led_config.yaml               ✅ exists
│   ├── pegasus_config.yaml           ✅ exists (corrected version below)
│   └── classifier_config.yaml        ✅ exists
├── outputs/                          ← model checkpoints, not committed
└── results/
    ├── token_length_distributions.png  ✅ generated
    └── domain_distribution.png         ✅ generated
```

---

## pyproject.toml (current working version)

```toml
[project]
name = "research-summarizer"
version = "1.0.0"
description = "Domain-Adaptive Multi-Model Research Paper Summarization System"
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
    "torch==2.5.1",
    "accelerate==0.34.2",
    "datasets==2.20.0",
    "tokenizers==0.20.3",
    "spacy==3.8.4",
    "nltk==3.9.1",
    "keybert==0.8.5",
    "yake==0.4.8",
    "pymupdf==1.24.10",
    "langchain==0.3.3",
    "langchain-community==0.3.2",
    "rouge-score==0.1.2",
    "bert-score==0.3.13",
    "scikit-learn==1.5.2",
    "wandb==0.18.3",
    "jupyter==1.1.1",
    "ipykernel==6.29.5",
    "matplotlib==3.9.2",
    "seaborn==0.13.2",
    "pandas==2.2.3",
    "numpy==1.26.4",
    "tqdm==4.66.5",
    "python-dotenv==1.0.1",
    "pyyaml==6.0.2",
    "transformers==4.46.3",
    "peft==0.13.2",
    "sentence-transformers==3.1.1",
    "pip>=26.1.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

---

## Config Files

### configs/pegasus_config.yaml (CORRECTED — previous version was wrong copy of LED config)

```yaml
# PEGASUS Fine-tuning Configuration
# google/pegasus-pubmed on ccdv/pubmed

model:
    name: "google/pegasus-pubmed"
    max_input_tokens: 1024
    max_output_tokens: 256

data:
    dataset_id: "ccdv/pubmed"
    input_field: "article"
    target_field: "abstract"

training:
    seed: 42
    epochs: 3
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 2 # effective batch size = 8
    per_device_eval_batch_size: 4
    optimizer: "adamw_torch"
    learning_rate: 3.0e-5
    lr_scheduler_type: "linear"
    warmup_steps: 500
    weight_decay: 0.01
    fp16: true
    evaluation_strategy: "epoch"
    save_strategy: "epoch"
    load_best_model_at_end: true
    metric_for_best_model: "eval_loss"

output:
    output_dir: "outputs/pegasus_finetuned"
    logging_dir: "outputs/pegasus_logs"

wandb:
    project: "research-summarizer"
    run_name: "pegasus-pubmed-finetune"
```

### configs/led_config.yaml

```yaml
# LED Fine-tuning Configuration
# allenai/led-large-16384 on ccdv/arxiv-summarization

model:
    name: "allenai/led-large-16384"
    max_input_tokens: 16384
    max_output_tokens: 512

data:
    dataset_id: "ccdv/arxiv-summarization"
    input_field: "article"
    target_field: "abstract"

training:
    seed: 42
    epochs: 3
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8 # effective batch size = 8
    per_device_eval_batch_size: 1
    optimizer: "adamw_torch"
    learning_rate: 3.0e-5
    lr_scheduler_type: "linear"
    warmup_steps: 500
    weight_decay: 0.01
    fp16: true
    evaluation_strategy: "epoch"
    save_strategy: "epoch"
    load_best_model_at_end: true
    metric_for_best_model: "eval_loss"

output:
    output_dir: "outputs/led_finetuned"
    logging_dir: "outputs/led_logs"

wandb:
    project: "research-summarizer"
    run_name: "led-large-arxiv-finetune"
```

### configs/classifier_config.yaml

```yaml
# BERT Domain Classifier Configuration
# bert-base-uncased fine-tuned for General vs Medical classification

model:
    name: "bert-base-uncased"
    max_input_tokens: 512
    num_labels: 2
    label_map:
        0: "General"
        1: "Medical"

data:
    arxiv_dataset_id: "ccdv/arxiv-summarization"
    pubmed_dataset_id: "ccdv/pubmed"
    samples_per_class: 10000
    input_fields: ["title", "abstract"]

training:
    seed: 42
    epochs: 3
    per_device_train_batch_size: 16
    gradient_accumulation_steps: 1
    per_device_eval_batch_size: 32
    optimizer: "adamw_torch"
    learning_rate: 2.0e-5
    lr_scheduler_type: "linear"
    warmup_ratio: 0.1
    weight_decay: 0.01
    fp16: true
    evaluation_strategy: "epoch"
    save_strategy: "epoch"
    load_best_model_at_end: true
    metric_for_best_model: "f1"

output:
    output_dir: "outputs/classifier"
    logging_dir: "outputs/classifier_logs"

wandb:
    project: "research-summarizer"
    run_name: "bert-domain-classifier"
```

---

## src/preprocessing.py (complete, working)

```python
"""
preprocessing.py
----------------
Reusable preprocessing pipeline for the Domain-Adaptive Research Paper
Summarization System.

Covers:
    - Text cleaning (regex + spaCy)
    - Section extraction (regex heuristics)
    - Sentence segmentation (spaCy)
    - Keyword extraction (KeyBERT + YAKE)
    - Stopword removal (NLTK + spaCy)
    - Named Entity Recognition (spaCy NER)

All functions are stateless and accept plain strings.
Call build_pipeline() once to load heavy models, then pass the returned
pipeline object into functions that need it.
"""

import re
import string
from typing import Optional

import nltk
import spacy
import yake
from keybert import KeyBERT
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

SECTION_PATTERNS = {
    "abstract"    : r"\b(abstract)\b",
    "introduction": r"\b(introduction|background)\b",
    "methodology" : r"\b(method(ology)?s?|approach|proposed\s+method|materials?\s+and\s+methods?)\b",
    "results"     : r"\b(results?|findings?|experiments?|experimental\s+results?)\b",
    "discussion"  : r"\b(discussion|analysis|interpretation)\b",
    "conclusion"  : r"\b(conclusion[s]?|summary|closing\s+remarks?|future\s+work)\b",
}

LATEX_PATTERNS = [
    r"\$[^$]*\$",
    r"\$\$[^$]*\$\$",
    r"\\[a-zA-Z]+\{[^}]*\}",
    r"\\[a-zA-Z]+",
    r"\{[^}]*\}",
    r"https?://\S+",
    r"\[\d+\]",
    r"\(\d{4}\)",
    r"et\s+al\.",
    r"fig\.\s*\d+",
    r"table\s*\d+",
    r"equation\s*\d+",
]

COMPILED_LATEX = [re.compile(p, re.IGNORECASE) for p in LATEX_PATTERNS]


class PreprocessingPipeline:
    def __init__(self, nlp, kw_model, yake_extractor, stop_words):
        self.nlp            = nlp
        self.kw_model       = kw_model
        self.yake_extractor = yake_extractor
        self.stop_words     = stop_words


def build_pipeline(keybert_model: str = "all-MiniLM-L6-v2") -> PreprocessingPipeline:
    print("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm")
    print("Loading KeyBERT model ...")
    kw_model = KeyBERT(model=keybert_model)
    print("Loading YAKE extractor ...")
    yake_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=10)
    stop_words = set(stopwords.words("english"))
    print("Pipeline ready.")
    return PreprocessingPipeline(nlp, kw_model, yake_extractor, stop_words)


def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    for pattern in COMPILED_LATEX:
        text = pattern.sub(" ", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(text: str) -> dict:
    sections = {name: "" for name in SECTION_PATTERNS}
    heading_positions = []
    for section_name, pattern in SECTION_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = match.start()
            preceding = text[max(0, start - 2): start]
            if preceding.strip() == "" or start == 0:
                heading_positions.append((start, section_name))
    heading_positions.sort(key=lambda x: x[0])
    for i, (pos, name) in enumerate(heading_positions):
        next_pos = (
            heading_positions[i + 1][0]
            if i + 1 < len(heading_positions)
            else len(text)
        )
        content_start = pos + len(name) + 1
        sections[name] = text[content_start:next_pos].strip()
    if all(v == "" for v in sections.values()):
        sections["introduction"] = text
    return sections


def get_body_text(sections: dict, preferred: Optional[list] = None) -> str:
    if preferred is None:
        preferred = ["methodology", "results", "introduction", "discussion", "conclusion"]
    for name in preferred:
        if sections.get(name, "").strip():
            return sections[name]
    return " ".join(v for v in sections.values() if v.strip())


def segment_sentences(text: str, pipeline: PreprocessingPipeline) -> list:
    if not text.strip():
        return []
    doc = pipeline.nlp(text[:100_000])
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def remove_stopwords(text: str, pipeline: PreprocessingPipeline) -> str:
    tokens = text.lower().split()
    filtered = [
        t for t in tokens
        if t not in pipeline.stop_words
        and t not in string.punctuation
        and len(t) > 1
    ]
    return " ".join(filtered)


def extract_keywords_keybert(text, pipeline, top_n=10, keyphrase_ngram_range=(1, 2)):
    if not text.strip():
        return []
    return pipeline.kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words="english",
        top_n=top_n,
    )


def extract_keywords_yake(text, pipeline, top_n=10):
    if not text.strip():
        return []
    return pipeline.yake_extractor.extract_keywords(text)[:top_n]


def extract_keywords_combined(text, pipeline, top_n=10):
    kb_keywords   = [kw for kw, _ in extract_keywords_keybert(text, pipeline, top_n=top_n)]
    yake_keywords = [kw for kw, _ in extract_keywords_yake(text, pipeline, top_n=top_n)]
    seen, combined = set(), []
    for kw in kb_keywords + yake_keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            combined.append(kw)
    return combined[:top_n]


def extract_entities(text: str, pipeline: PreprocessingPipeline) -> dict:
    if not text.strip():
        return {}
    doc = pipeline.nlp(text[:100_000])
    entities: dict = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text.strip() not in entities[ent.label_]:
            entities[ent.label_].append(ent.text.strip())
    return entities


def preprocess(text: str, pipeline: PreprocessingPipeline) -> dict:
    cleaned   = clean_text(text)
    sections  = extract_sections(cleaned)
    sentences = segment_sentences(cleaned, pipeline)
    no_sw     = remove_stopwords(cleaned, pipeline)
    keywords  = extract_keywords_combined(cleaned, pipeline, top_n=10)
    entities  = extract_entities(cleaned, pipeline)
    return {
        "cleaned_text" : cleaned,
        "sections"     : sections,
        "sentences"    : sentences,
        "no_stopwords" : no_sw,
        "keywords"     : keywords,
        "entities"     : entities,
    }
```

---

## Complete Implementation Plan

### ✅ Phase 0 — Project Scaffolding (COMPLETE)

- `pyproject.toml`, `README.md`, `.gitignore` created
- Full folder structure created
- `uv` environment set up with Python 3.11
- All config YAML files created
- `.env` file set up with `HF_DATASETS_CACHE` and `HF_HOME`

### ✅ Phase 1 — Data Loading & EDA (COMPLETE)

**Notebook:** `01_data_loading_and_eda.ipynb`

- Dual-mode loader built (`"dev"` / `"full"`) — switch via single flag
- Dev dataset (Devika03) loaded, 330 rows, split 80/10/10
- Unified schema normalised: `article`, `summary`, `domain`, `title`, `keywords`
- Split sizes verified, null check passed
- Token length distributions plotted and saved
- Domain distribution plotted (259 General / 71 Medical in dev)
- Processed splits saved to `data/processed_dev/`

**Key stats (dev mode):**

- Articles: mean=498 tokens, median=464, p95=839, max=1,148
- Summaries: mean=138 tokens, median=132, p95=199, max=310

### ✅ Phase 2 — Preprocessing Pipeline (COMPLETE)

**Notebook:** `02_preprocessing_pipeline.ipynb`
**Source:** `src/preprocessing.py`

- Text cleaning (LaTeX, URLs, citations, non-ASCII)
- Section extraction (regex heuristics for 6 section types)
- Sentence segmentation (spaCy `en_core_web_sm`)
- Stopword removal (NLTK)
- Keyword extraction (KeyBERT + YAKE combined)
- Named Entity Recognition (spaCy NER)
- Full pipeline runs on all 264 train rows
- Preprocessed data saved to `data/preprocessed_dev/train_preprocessed.json`

**Note on dev data:** Section extraction finds mostly `introduction` only
because Devika03 articles are abstract+conclusion concatenations without
explicit section headers. On full arXiv/PubMed data, all 6 sections will
be found correctly.

---

### 🔲 Phase 3 — Domain Classifier (NEXT)

**Notebook:** `03_domain_classifier.ipynb`
**Source:** `src/classifier.py`
**SLURM:** `slurm/train_classifier.sbatch`

Tasks:

1. Load processed splits from `data/processed_{DATASET_MODE}/`
2. Build classifier dataset:
    - Dev mode: use `domain` field from processed JSON (General=0, Medical=1)
    - Full mode: sample 10K from each of arXiv (General) and PubMed (Medical)
3. Tokenize input as `title + " [SEP] " + article[:512_tokens]` using
   `bert-base-uncased` tokenizer
4. Fine-tune `bert-base-uncased` with classification head using HuggingFace
   `Trainer` API — config from `configs/classifier_config.yaml`
5. Set `seed=42` everywhere, log all metrics to wandb
6. Evaluate: accuracy, F1-score, confusion matrix
7. Save checkpoint to `outputs/classifier/`
8. Write `src/classifier.py` with `predict_domain(text) -> (label, confidence)`
9. Write `slurm/train_classifier.sbatch` with GPU allocation, memory limits,
   wandb env export, and `# CLUSTER-SPECIFIC` placeholders for module loading

**Dev mode note:** Only 330 rows total with class imbalance (259 General /
71 Medical). Classifier will train but metrics won't be meaningful. This is
expected — full mode gives 10K per class. Still write and run it to verify
the code works end-to-end.

---

### 🔲 Phase 4 — LED Fine-tuning (General Domain)

**Notebook:** `04_led_finetuning.ipynb`
**SLURM:** `slurm/train_led_arxiv.sbatch`

Tasks:

1. Load `allenai/led-large-16384` and tokenizer
2. Prepare arXiv dataset — tokenize `article` (up to 16,384 tokens),
   `abstract` as target
3. Training config from `configs/led_config.yaml`:
    - 3 epochs, batch=1, grad accumulation=8, lr=3e-5, linear warmup,
      fp16, seed=42
4. Fine-tune with HuggingFace `Trainer` + `accelerate`
5. Log training/validation loss curves to wandb
6. Show sample output at each epoch checkpoint
7. Save best checkpoint to `outputs/led_finetuned/`
8. Write `slurm/train_led_arxiv.sbatch` — **submit this first, it's the
   longest job (~6–8 hrs)**
9. **Justify in notebook markdown:** LED chosen over BART because BART
   truncates at 1,024 tokens. LED handles 16,384 — full paper length.

**Dev mode:** Run on dev data just to confirm the training loop works.
Flip to `"full"` on HPC for actual fine-tuning.

---

### 🔲 Phase 5 — PEGASUS Fine-tuning (Medical Domain)

**Notebook:** `05_pegasus_finetuning.ipynb`
**SLURM:** `slurm/train_pegasus_pubmed.sbatch`

Tasks:

1. Load `google/pegasus-pubmed` (already domain-pretrained on PubMed)
2. Prepare PubMed dataset — tokenize `article` and `abstract`
3. Training config from `configs/pegasus_config.yaml`:
    - 3 epochs, batch=4, grad accumulation=2, lr=3e-5, linear warmup,
      fp16, seed=42
4. Fine-tune with HuggingFace `Trainer` API; log to wandb
5. Show sample outputs per epoch
6. Save best checkpoint to `outputs/pegasus_finetuned/`
7. Write `slurm/train_pegasus_pubmed.sbatch`

**HPC submission:** Submit simultaneously with `train_led_arxiv.sbatch`.
While both run, write the report.

---

### 🔲 Phase 6 — Evaluation & Angle 1 Benchmark

**Notebook:** `06_evaluation_and_comparison.ipynb`
**SLURM:** `slurm/evaluate_all.sbatch`
**Source:** `src/evaluation.py`

Tasks:

1. Implement ROUGE-1/2/L scoring using `rouge-score`
2. Implement BERTScore-F1 using `bert-score`
3. Run all 6 benchmark rows (this is the core Angle 1 result):

| Model              | Test Domain                |
| ------------------ | -------------------------- |
| LED zero-shot      | General (arXiv test set)   |
| LED fine-tuned     | General (arXiv test set)   |
| PEGASUS zero-shot  | Medical (PubMed test set)  |
| PEGASUS fine-tuned | Medical (PubMed test set)  |
| LED fine-tuned     | **Medical (cross-domain)** |
| PEGASUS fine-tuned | **General (cross-domain)** |

4. Cross-domain rows are the empirical proof that routing is necessary —
   highlight prominently in the report
5. Save all scores to `results/rouge_scores.json` and `results/bertscores.json`
6. Write `slurm/evaluate_all.sbatch` — submit after training jobs complete

---

### 🔲 Phase 7 — Domain Routing System (Angle 2)

**Notebook:** `07_routing_system.ipynb`
**Source:** `src/router.py`

Tasks:

1. Load fine-tuned BERT classifier from `outputs/classifier/`
2. Implement router: input → `predict_domain()` → LED (General) or
   PEGASUS (Medical)
3. Wire full end-to-end pipeline: input → classifier → summarizer → output
4. Demo on 5+ unseen papers
5. Show classifier confidence scores alongside summaries
6. Validate routing matches Angle 1 benchmark results

---

### 🔲 Phase 8 — Structured Output Generator

**Notebook:** `08_structured_output_prompting.ipynb`
**Source:** `src/output_generator.py`

Tasks:

1. Set up LLM layer (Llama local on HPC, or API) via LangChain
2. Implement 3 prompt variants for ablation:
    - Zero-shot: single instruction, no examples
    - Few-shot (3-shot): 3 example input-output pairs prepended
    - Chain-of-thought: step-by-step reasoning before output
3. Each variant outputs this exact JSON schema:

```json
{
    "title": "string",
    "domain": "General | Medical",
    "confidence": 0.95,
    "summary": "2-3 sentence summary",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "methodology_overview": "string",
    "limitations": ["limitation 1", "limitation 2"],
    "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
    "citation_apa": "Author, A. (Year). Title. Journal. DOI"
}
```

4. Run all 3 variants on same papers, compare output quality
5. Report prompt ablation results

---

### 🔲 Phase 9 — Human Evaluation

**Notebook:** `09_human_evaluation.ipynb`

Tasks:

1. Select 20–30 summaries from both domains and multiple models
2. Human evaluation rubric (5 criteria, 1–5 scale each):
    - Accuracy, Conciseness, Coherence, Completeness, Domain Relevance
3. Collect ratings from at least 2 annotators
4. Calculate inter-annotator agreement using Cohen's Kappa (`scikit-learn`)
5. Save ratings to `results/human_eval.csv`
6. Report mean scores per model and per criterion

---

### 🔲 Phase 10 — Integration & Final Review

1. Every `src/*.py` function has full docstring (args, returns, description)
2. All notebooks have markdown cells before each major code block
3. All magic numbers are in config YAMLs, never hardcoded mid-cell
4. `seed = 42` set consistently everywhere
5. All training scripts log to wandb
6. README covers: setup, notebook run order, SLURM submission, reproduction

---

## HPC SLURM Job Submission Order

```bash
# Day 1 on HPC — submit simultaneously (longest jobs)
sbatch slurm/train_led_arxiv.sbatch
sbatch slurm/train_pegasus_pubmed.sbatch

# After data prep is confirmed working
sbatch slurm/train_classifier.sbatch

# After ALL training jobs complete
sbatch slurm/evaluate_all.sbatch

# Check job status
squeue -u $USER
```

---

## Code Quality Rules (Enforced Throughout)

- Every function: docstring with Args, Returns, one-line description
- Every notebook: markdown cell before each major code block
- All hyperparameters: defined in config YAMLs or at top of notebook
- Random seed: `SEED = 42` set at top of every notebook and script
- Logging: every training script logs to wandb
- Package manager: `uv` only — never `pip install` directly

---

## Notes for Next Chat

- Do NOT revisit LED vs BART decision — LED is confirmed and justified
- Do NOT use pip — always use `uv add`
- Always start notebooks with `DATASET_MODE = "dev"` for local testing
- The `pegasus_config.yaml` had a copy-paste error (was a copy of LED config)
  — it has been corrected in this document; update the actual file before
  continuing
- `data/preprocessed_dev/train_preprocessed.json` exists with all 264 train
  rows fully preprocessed
- When switching to HPC, run Notebook 01 first with `DATASET_MODE = "full"`
  before anything else
- Start next session from **Phase 3 — Domain Classifier**
