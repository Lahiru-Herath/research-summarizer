# Domain-Adaptive Multi-Model Research Paper Summarization System
### Advanced AI – Group Project Architecture Document

---

## 1. Project Overview

This project proposes a **Domain-Adaptive Research Paper Summarization System** that combines two core research contributions:

- **Angle 1 — Comparative Evaluation Framework:** A rigorous empirical benchmark comparing multiple pretrained and fine-tuned summarization models across general and medical scientific domains, using standardized quantitative metrics.
- **Angle 2 — Domain-Aware Model Routing:** A BERT-based domain classifier that automatically routes each input paper to its best-performing domain-specific summarization model, validated by the benchmark results from Angle 1.

The system takes a research paper as input and produces a structured, domain-optimized summary — with every design decision backed by quantitative evidence.

---

## 2. Datasets

| Dataset | Source | Size | Domain | Purpose |
|---|---|---|---|---|
| `ccdv/arxiv-summarization` | HuggingFace | ~215,000 pairs | General AI / CS / Physics | Fine-tune general summarizer; Angle 1 evaluation |
| `ccdv/pubmed` | HuggingFace | ~119,000 pairs | Biomedical / Medical | Fine-tune medical summarizer; Angle 1 evaluation |

### Dataset Fields Used

**arXiv:**
- `article` → input (full paper body)
- `abstract` → target summary (ground truth)

**PubMed:**
- `article` → input (full paper body)
- `abstract` → target summary (ground truth)

### Dataset Splits

Both datasets come with pre-defined splits:

| Split | arXiv | PubMed |
|---|---|---|
| Train | 203,037 | 119,924 |
| Validation | 6,436 | 6,633 |
| Test | 6,440 | 6,658 |

---

## 3. System Architecture

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
│                                                                 │
│         Fine-tuned BERT (bert-base-uncased)                     │
│         Trained on topic labels from both datasets              │
│                                                                 │
│         Classes:                                                │
│         ├── General (AI / CS / Physics / Engineering)           │
│         └── Medical (Biomedical / Clinical / Healthcare)        │
│                                                                 │
│         Output: Domain Label + Confidence Score                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
┌─────────────────────┐   ┌─────────────────────────┐
│   GENERAL DOMAIN    │   │     MEDICAL DOMAIN      │
│                     │   │                         │
│  facebook/          │   │  google/                │
│  bart-large-cnn     │   │  pegasus-pubmed         │
│                     │   │                         │
│  Fine-tuned on      │   │  Fine-tuned on          │
│  ccdv/arxiv-        │   │  ccdv/pubmed            │
│  summarization      │   │                         │
└──────────┬──────────┘   └────────────┬────────────┘
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               STRUCTURED OUTPUT GENERATOR                       │
│                                                                 │
│   Prompt-engineered LLM layer (Llama / GPT via API)            │
│                                                                 │
│   ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│   │ Concise      │  │ Key Findings   │  │ Methodology      │   │
│   │ Summary      │  │ (bullet list)  │  │ Overview         │   │
│   └──────────────┘  └────────────────┘  └──────────────────┘   │
│                                                                 │
│   ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│   │ Limitations  │  │ Keywords       │  │ APA Citation     │   │
│   │ & Gaps       │  │ (top 5–10)     │  │ (auto-generated) │   │
│   └──────────────┘  └────────────────┘  └──────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                             │
│                                                                 │
│   Quantitative:                                                 │
│   ├── ROUGE-1, ROUGE-2, ROUGE-L                                 │
│   ├── BERTScore (Precision, Recall, F1)                         │
│   └── Model Comparison Table (Angle 1 benchmark)               │
│                                                                 │
│   Qualitative:                                                  │
│   ├── Human Evaluation Rubric (5 criteria, 1–5 scale)          │
│   └── Inter-Annotator Agreement (Cohen's Kappa)                │
│                                                                 │
│   Ablation Studies:                                             │
│   ├── Zero-shot vs Fine-tuned (transfer learning delta)         │
│   ├── General model on medical papers vs domain-specific model  │
│   └── Generic prompt vs Chain-of-thought prompt                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Components in Detail

### 4.1 Preprocessing Pipeline

| Step | Tool / Library | Output |
|---|---|---|
| PDF parsing | PyMuPDF (`fitz`) | Raw text |
| Text cleaning | regex + spaCy | Cleaned text |
| Sentence segmentation | spaCy (`en_core_web_sm`) | Sentence list |
| Section extraction | regex heuristics | Structured sections dict |
| Keyword extraction | KeyBERT / YAKE | Top N keywords |
| Stopword removal | NLTK / spaCy | Filtered tokens |
| Named Entity Recognition | spaCy NER | Entities (drugs, methods, etc.) |

---

### 4.2 Domain Classifier

| Property | Detail |
|---|---|
| Base model | `bert-base-uncased` |
| Fine-tuning data | Topic/category labels from arXiv + PubMed metadata |
| Classes | General, Medical |
| Input | Paper title + abstract (first 512 tokens) |
| Output | Domain label + confidence score |
| Training | HuggingFace `Trainer` API |
| Evaluation metric | Accuracy, F1-score |

---

### 4.3 Summarization Models

#### General Domain Model
| Property | Detail |
|---|---|
| Base model | `facebook/bart-large-cnn` |
| Fine-tuning dataset | `ccdv/arxiv-summarization` (full 203K train set) |
| Input max tokens | 1,024 |
| Output max tokens | 256 |
| Training epochs | 3 |
| Batch size | 8 (with gradient accumulation × 4) |
| Optimizer | AdamW, lr = 3e-5 |
| Scheduler | Linear warmup |

#### Medical Domain Model
| Property | Detail |
|---|---|
| Base model | `google/pegasus-pubmed` |
| Fine-tuning dataset | `ccdv/pubmed` (full 119K train set) |
| Input max tokens | 1,024 |
| Output max tokens | 256 |
| Training epochs | 3 |
| Batch size | 8 (with gradient accumulation × 4) |
| Optimizer | AdamW, lr = 3e-5 |
| Scheduler | Linear warmup |

---

### 4.4 Structured Output Generator

Uses **prompt engineering** and **few-shot in-context learning** to transform the raw model summary into a structured document.

#### Prompt Strategy Variants (for Ablation)

| Variant | Description |
|---|---|
| Zero-shot | Single instruction prompt, no examples |
| Few-shot (3-shot) | 3 example input-output pairs prepended to prompt |
| Chain-of-thought | Step-by-step reasoning instructions before output |

#### Output Schema

```json
{
  "title": "string",
  "domain": "General | Medical",
  "summary": "string (2–3 sentences)",
  "key_findings": ["string", "string", "string"],
  "methodology_overview": "string",
  "limitations": ["string", "string"],
  "keywords": ["string", "string", "string"],
  "citation_apa": "string"
}
```

---

### 4.5 Evaluation Layer

#### Quantitative Metrics

| Metric | Library | What it Measures |
|---|---|---|
| ROUGE-1 | `rouge-score` | Unigram overlap with reference |
| ROUGE-2 | `rouge-score` | Bigram overlap with reference |
| ROUGE-L | `rouge-score` | Longest common subsequence |
| BERTScore-F1 | `bert-score` | Semantic similarity to reference |

#### Angle 1 — Model Comparison Table (Target Output)

| Model | Domain | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|---|---|---|---|---|---|
| BART (zero-shot) | General | - | - | - | - |
| BART (fine-tuned) | General | - | - | - | - |
| PEGASUS (zero-shot) | Medical | - | - | - | - |
| PEGASUS (fine-tuned) | Medical | - | - | - | - |
| BART fine-tuned on Medical | Medical | - | - | - | - |
| PEGASUS fine-tuned on General | General | - | - | - | - |

> The last two rows are cross-domain tests — feeding a general model medical papers and vice versa. This is the empirical proof that domain routing (Angle 2) is necessary.

#### Qualitative — Human Evaluation Rubric

| Criterion | Description | Scale |
|---|---|---|
| Accuracy | Does the summary correctly reflect the paper? | 1–5 |
| Conciseness | Is it free of redundant information? | 1–5 |
| Coherence | Is the summary well-structured and readable? | 1–5 |
| Completeness | Are all major contributions captured? | 1–5 |
| Domain Relevance | Is the vocabulary appropriate for the domain? | 1–5 |

Inter-annotator agreement measured using **Cohen's Kappa**.

---

## 5. Technical Stack

| Layer | Tools / Libraries |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Fine-tuning | HuggingFace `Trainer` API, PEFT (optional LoRA) |
| NLP Preprocessing | spaCy, NLTK, KeyBERT, YAKE |
| PDF Parsing | PyMuPDF (`fitz`) |
| Agent Orchestration | LangChain |
| Evaluation | `rouge-score`, `bert-score`, `scikit-learn` |
| Experiment Tracking | Weights & Biases (wandb) |
| Notebooks | Jupyter (one per major component) |
| HPC Job Scheduling | SLURM (`.sbatch` scripts per training job) |
| Version Control | Git + GitHub |

---

## 6. HPC Job Strategy

Submit these jobs in parallel from day one:

| Job | Script | Estimated Time | Priority |
|---|---|---|---|
| Fine-tune BART on arXiv | `train_bart_arxiv.sbatch` | ~6–8 hrs | Highest — submit first |
| Fine-tune PEGASUS on PubMed | `train_pegasus_pubmed.sbatch` | ~6–8 hrs | Highest — submit first |
| Fine-tune BERT classifier | `train_classifier.sbatch` | ~1–2 hrs | Medium |
| Evaluation runs | `evaluate_all.sbatch` | ~2–3 hrs | Submit after training jobs |

**While HPC jobs run → write the report.**

---

## 7. Notebook Structure

| Notebook | Content |
|---|---|
| `01_data_loading_and_eda.ipynb` | Load datasets, explore distributions, sample visualization |
| `02_preprocessing_pipeline.ipynb` | Text cleaning, section extraction, keyword extraction |
| `03_domain_classifier.ipynb` | BERT fine-tuning for domain classification |
| `04_bart_finetuning.ipynb` | BART fine-tuning on arXiv (or reference to SLURM script) |
| `05_pegasus_finetuning.ipynb` | PEGASUS fine-tuning on PubMed (or reference to SLURM script) |
| `06_evaluation_and_comparison.ipynb` | Angle 1 — ROUGE, BERTScore, comparison table |
| `07_routing_system.ipynb` | Angle 2 — classifier + router + end-to-end demo |
| `08_structured_output_prompting.ipynb` | Prompt engineering ablation study |
| `09_human_evaluation.ipynb` | Human rubric scoring, Cohen's Kappa calculation |

---

## 8. Required AI Techniques Coverage

| Assignment Requirement | How It Is Covered |
|---|---|
| NLP — text preprocessing | spaCy pipeline: tokenization, NER, stopwords, section extraction |
| NLP — word embeddings | KeyBERT uses sentence-transformers embeddings for keyword extraction |
| Transformer-based LLMs | BART and PEGASUS as core summarization models |
| Fine-tuning / Transfer Learning | Pretrained checkpoints fine-tuned on domain-specific datasets |
| Prompt Engineering | Structured output generator: zero-shot vs few-shot vs chain-of-thought ablation |
| Few-shot Learning | 3-shot prompting in structured output stage |

---

## 9. Expected Contributions

1. **Empirical benchmark** (Angle 1) showing quantitative performance differences between general and domain-specific summarization models on scientific papers — filling a gap that existing tools like NotebookLM do not address publicly.

2. **Domain-adaptive routing system** (Angle 2) validated by benchmark evidence — a BERT classifier that improves end-to-end summarization quality by routing papers to their optimal model.

3. **Reproducible pipeline** on two large-scale public datasets with full evaluation code, enabling future researchers to extend the benchmark to additional domains.

---

## 10. Future Work (Angles 3 & 4)

- **Angle 3 — Multi-document Synthesis:** Extend the system to synthesize across multiple papers on the same topic, identifying consensus findings, contradictions, and research gaps.
- **Angle 4 — Richer Structured Outputs:** Generate literature-review-ready output sections, comparative methodology tables across papers, and citation graph visualizations.

---

*Document prepared for Advanced Artificial Intelligence – Group Project*
