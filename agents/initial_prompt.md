# Development Prompt — Domain-Adaptive Research Paper Summarization System

## Context

You are helping me build a complete, production-quality AI research project for an Advanced Artificial Intelligence university course. This is a fully scoped project with an HPC cluster available, so there are no computational constraints. We will fine-tune models, run full evaluations, and produce well-documented Jupyter notebooks. Do not simplify or suggest shortcuts unless explicitly asked.

---

## Project Summary

We are building a **Domain-Adaptive Multi-Model Research Paper Summarization System** with two core research contributions:

- **Angle 1 — Comparative Evaluation Framework:** A rigorous empirical benchmark comparing multiple pretrained and fine-tuned summarization models across general and medical scientific domains using ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore.
- **Angle 2 — Domain-Aware Model Routing:** A fine-tuned BERT-based domain classifier that automatically routes each input paper to its best-performing domain-specific summarization model, validated by the Angle 1 benchmark results.

The cross-domain rows in the evaluation (running the general model on medical papers and vice versa) are critical — they are the empirical proof that domain routing is necessary and not just a design preference.

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│         PDF / ArXiv Link / DOI / Raw Text / PubMed ID          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                        │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Text        │  │ Section      │  │ Keyword               │  │
│  │ Cleaning    │→ │ Extraction   │→ │ Extraction            │  │
│  │ (spaCy)     │  │ (regex +     │  │ (KeyBERT / YAKE)      │  │
│  │             │  │  heuristics) │  │                       │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
│                                                                 │
│  Extracted Sections: Abstract | Introduction | Methodology |    │
│                      Results  | Discussion   | Conclusion       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DOMAIN CLASSIFIER                             │
│         Fine-tuned BERT (bert-base-uncased)                     │
│         Classes: General | Medical                              │
│         Input: Title + Abstract (first 512 tokens)             │
│         Output: Domain Label + Confidence Score                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
┌─────────────────────┐   ┌─────────────────────────┐
│   GENERAL DOMAIN    │   │     MEDICAL DOMAIN      │
│                     │   │                         │
│  allenai/           │   │  google/                │
│  led-large-16384    │   │  pegasus-pubmed         │
│                     │   │                         │
│  Fine-tuned on      │   │  Fine-tuned on          │
│  ccdv/arxiv-        │   │  ccdv/pubmed            │
│  summarization      │   │                         │
│                     │   │                         │
│  Handles up to      │   │  Domain-pretrained,     │
│  16,384 tokens —    │   │  purpose-built for      │
│  full paper length  │   │  biomedical text        │
└──────────┬──────────┘   └────────────┬────────────┘
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               STRUCTURED OUTPUT GENERATOR                       │
│   Prompt-engineered LLM layer (Llama local on HPC or API)      │
│                                                                 │
│   Outputs: Summary | Key Findings | Methodology Overview |      │
│            Limitations | Keywords | APA Citation               │
│                                                                 │
│   Prompt variants (ablation): Zero-shot | Few-shot | CoT        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                             │
│   ROUGE-1/2/L, BERTScore, Human Evaluation, Cohen's Kappa      │
│   Ablation: zero-shot vs fine-tuned, cross-domain tests,        │
│             prompt variant comparison                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Datasets

| Dataset | HuggingFace ID | Size | Domain | Usage |
|---|---|---|---|---|
| arXiv Summarization | `ccdv/arxiv-summarization` | ~215K pairs | General (AI/CS/Physics) | Fine-tune LED; Angle 1 general evaluation |
| PubMed Summarization | `ccdv/pubmed` | ~119K pairs | Biomedical / Medical | Fine-tune PEGASUS; Angle 1 medical evaluation |

Both datasets have pre-defined train/validation/test splits. Use them as-is — do not re-split.

- `article` field = input (full paper body)
- `abstract` field = target summary (ground truth)

---

## Models

### General Domain — `allenai/led-large-16384` (Longformer Encoder-Decoder)
- Handles up to 16,384 tokens — critical for full research papers
- Architecturally superior to BART for long document summarization
- Fine-tune on `ccdv/arxiv-summarization`
- We chose this over `facebook/bart-large-cnn` because BART truncates at 1,024 tokens (losing most of a research paper) and is notoriously sensitive to hyperparameters during fine-tuning
- This choice should be explicitly justified in the report methodology section

### Medical Domain — `google/pegasus-pubmed`
- Already pretrained on PubMed abstracts — domain-pretrained checkpoint
- Fine-tune further on `ccdv/pubmed` for additional adaptation
- Purpose-built for biomedical summarization

### Domain Classifier — `bert-base-uncased`
- Fine-tuned to classify papers as General or Medical
- Input: paper title + abstract (concatenated, max 512 tokens)
- Output: domain label + confidence score
- Training data: derive labels from dataset source (arXiv = General, PubMed = Medical)

---

## Package Manager

**Use `uv` exclusively.** Do not use pip directly. All dependency installation commands must use `uv`. Use `uv` for creating virtual environments, adding packages, and running scripts.

Example:
```bash
uv venv
source .venv/bin/activate
uv add transformers datasets torch accelerate
```

Provide a `pyproject.toml` (not `requirements.txt`) as the dependency file.

---

## Technical Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Package Manager | `uv` (not pip) |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Fine-tuning | HuggingFace `Trainer` API + `accelerate` |
| Efficient Fine-tuning | PEFT / LoRA (optional, use if memory is tight) |
| NLP Preprocessing | spaCy (`en_core_web_sm`), NLTK, KeyBERT, YAKE |
| PDF Parsing | PyMuPDF (`fitz`) |
| Agent Orchestration | LangChain |
| Evaluation | `rouge-score`, `bert-score`, `scikit-learn` |
| Experiment Tracking | Weights & Biases (`wandb`) |
| Notebooks | Jupyter |
| HPC Job Scheduling | SLURM (provide `.sbatch` scripts for all training jobs) |
| Version Control | Git (provide `.gitignore`) |

---

## HPC Strategy

All major training jobs must have corresponding SLURM `.sbatch` scripts. Submit training jobs in parallel:

| Job | Script Name | Submit Order |
|---|---|---|
| Fine-tune LED on arXiv | `train_led_arxiv.sbatch` | First — longest job |
| Fine-tune PEGASUS on PubMed | `train_pegasus_pubmed.sbatch` | First — submit simultaneously |
| Fine-tune BERT classifier | `train_classifier.sbatch` | Second |
| Run all evaluations | `evaluate_all.sbatch` | After training completes |

SLURM scripts must include: GPU allocation, memory limits, wandb environment variable export, and proper module loading sections (with comments where cluster-specific values need to be filled in).

---

## Notebook Structure

Produce one Jupyter notebook per component. Each notebook must have markdown cells explaining every major step — not just code comments. This is a graded submission requirement.

| Notebook | Purpose |
|---|---|
| `01_data_loading_and_eda.ipynb` | Load both datasets, explore distributions, visualize summary length histograms, sample 5 examples |
| `02_preprocessing_pipeline.ipynb` | Text cleaning, section extraction, keyword extraction, NER — show before/after examples |
| `03_domain_classifier.ipynb` | BERT fine-tuning for binary domain classification, confusion matrix, F1 report |
| `04_led_finetuning.ipynb` | LED fine-tuning on arXiv, training curves, sample output at each epoch |
| `05_pegasus_finetuning.ipynb` | PEGASUS fine-tuning on PubMed, training curves, sample output |
| `06_evaluation_and_comparison.ipynb` | **Angle 1** — full ROUGE + BERTScore comparison table including cross-domain tests |
| `07_routing_system.ipynb` | **Angle 2** — classifier + router + end-to-end demo on unseen papers |
| `08_structured_output_prompting.ipynb` | Prompt engineering ablation: zero-shot vs few-shot vs chain-of-thought |
| `09_human_evaluation.ipynb` | Human rubric scoring template, Cohen's Kappa calculation |

---

## Evaluation Requirements

### Angle 1 — Model Comparison Table
This table must include cross-domain rows. It is the core result of the paper.

| Model | Test Domain | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|---|
| LED zero-shot | General | | | | |
| LED fine-tuned | General | | | | |
| PEGASUS zero-shot | Medical | | | | |
| PEGASUS fine-tuned | Medical | | | | |
| LED fine-tuned | Medical (cross-domain) | | | | |
| PEGASUS fine-tuned | General (cross-domain) | | | | |

### Qualitative — Human Evaluation Rubric (per summary, rated 1–5)
- Accuracy
- Conciseness
- Coherence
- Completeness
- Domain Relevance

Report inter-annotator agreement using Cohen's Kappa.

### Ablation Studies
1. Zero-shot vs fine-tuned (quantifies transfer learning contribution)
2. Cross-domain mismatches (justifies Angle 2 routing system)
3. Prompt variant comparison for structured output stage

---

## Structured Output Schema

The structured output generator must produce JSON in this exact schema:

```json
{
  "title": "string",
  "domain": "General | Medical",
  "confidence": 0.95,
  "summary": "2–3 sentence concise summary",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "methodology_overview": "string",
  "limitations": ["limitation 1", "limitation 2"],
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
  "citation_apa": "Author, A. (Year). Title. Journal. DOI"
}
```

---

## Project Folder Structure

Provide and maintain this exact structure:

```
research-summarizer/
├── pyproject.toml
├── README.md
├── .gitignore
├── data/
│   └── (datasets downloaded via HuggingFace, not committed to git)
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_domain_classifier.ipynb
│   ├── 04_led_finetuning.ipynb
│   ├── 05_pegasus_finetuning.ipynb
│   ├── 06_evaluation_and_comparison.ipynb
│   ├── 07_routing_system.ipynb
│   ├── 08_structured_output_prompting.ipynb
│   └── 09_human_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── classifier.py
│   ├── summarizer.py
│   ├── router.py
│   ├── output_generator.py
│   └── evaluation.py
├── slurm/
│   ├── train_led_arxiv.sbatch
│   ├── train_pegasus_pubmed.sbatch
│   ├── train_classifier.sbatch
│   └── evaluate_all.sbatch
├── configs/
│   ├── led_config.yaml
│   ├── pegasus_config.yaml
│   └── classifier_config.yaml
├── outputs/
│   └── (model checkpoints, results — not committed to git)
└── results/
    ├── rouge_scores.json
    ├── bertscores.json
    └── human_eval.csv
```

---

## Code Quality Requirements

These are graded submission requirements — enforce them throughout:

- Every function must have a docstring with args, returns, and a one-line description
- Every notebook must have markdown cells before each major code block explaining what it does and why
- All magic numbers (learning rates, batch sizes, max lengths) must be defined in config files or at the top of the notebook — never hardcoded mid-cell
- All random seeds must be set explicitly and consistently (`seed = 42` everywhere)
- Every training script must log to `wandb`
- The README must include: project description, setup instructions using `uv`, how to run each notebook, how to submit SLURM jobs, and how to reproduce results

---

## Assignment Techniques Coverage (Must Be Explicit in Code and Report)

| Technique | Where It Appears |
|---|---|
| NLP — text preprocessing | `02_preprocessing_pipeline.ipynb`, `src/preprocessing.py` |
| NLP — word embeddings | KeyBERT in preprocessing (uses sentence-transformers embeddings) |
| Transformer LLMs | LED and PEGASUS in notebooks 04 and 05 |
| Fine-tuning / Transfer learning | Fine-tuning both models; zero-shot vs fine-tuned ablation quantifies the delta |
| Prompt engineering | `08_structured_output_prompting.ipynb` — zero-shot vs few-shot vs CoT ablation |
| Few-shot learning | 3-shot prompting in structured output stage |

---

## What to Build First (Suggested Order)

1. `pyproject.toml` + `README.md` + `.gitignore` + folder structure
2. `01_data_loading_and_eda.ipynb` — verify datasets load correctly before anything else
3. `02_preprocessing_pipeline.ipynb` + `src/preprocessing.py`
4. `03_domain_classifier.ipynb` + `slurm/train_classifier.sbatch`
5. `04_led_finetuning.ipynb` + `slurm/train_led_arxiv.sbatch`
6. `05_pegasus_finetuning.ipynb` + `slurm/train_pegasus_pubmed.sbatch`
7. `06_evaluation_and_comparison.ipynb` — Angle 1
8. `07_routing_system.ipynb` — Angle 2
9. `08_structured_output_prompting.ipynb`
10. `09_human_evaluation.ipynb`

---

## Important Decisions Already Made — Do Not Revisit Unless Asked

- **LED over BART:** We explicitly chose `allenai/led-large-16384` over `facebook/bart-large-cnn` because BART truncates at 1,024 tokens and is hard to fine-tune. LED handles 16K tokens and is more stable. This is a deliberate, justified design decision.
- **uv over pip:** All package management uses `uv`. No pip commands anywhere.
- **Full dataset training:** We are using the full training sets (not subsamples) because HPC is available.
- **Two datasets:** `ccdv/arxiv-summarization` for general, `ccdv/pubmed` for medical. Both are used.
- **Angles 1 and 2 only:** Multi-document synthesis (Angle 3) and richer structured outputs (Angle 4) are reserved as future work in the report conclusion. Do not implement them.

---

*This prompt contains the complete specification. Build everything according to this document. Ask for clarification only if something is genuinely ambiguous — do not ask for confirmation on decisions already made above.*
