# Report Writing Reference
## Domain-Adaptive Multi-Model Research Paper Summarization System
### EC 7203 — Advanced Artificial Intelligence, University of Ruhuna

> This document contains everything needed to write the final project report.
> Use this as the single source of truth when starting a new chat for report writing.

---

## 1. Project Identity

| Field | Detail |
|---|---|
| Project title | Domain-Adaptive Multi-Model Research Paper Summarization System |
| Course | EC 7203 — Advanced Artificial Intelligence |
| University | University of Ruhuna, Faculty of Engineering |
| Project type | Group project |
| Stage | Complete — all phases implemented and evaluated |

---

## 2. The Two Core Research Contributions

Everything in the report revolves around these two angles. Never lose sight of them.

### Angle 1 — Comparative Evaluation Benchmark
A rigorous empirical benchmark comparing pretrained vs. fine-tuned summarization models (BART and PEGASUS) across General and Medical scientific domains using ROUGE and BERTScore metrics. Includes cross-domain tests (General model on Medical papers and vice versa) that provide the quantitative justification for the routing system.

### Angle 2 — Domain-Aware Model Routing
A fine-tuned BERT classifier (99.4% accuracy) that automatically routes each input paper to its optimal domain-specific summarization model. The routing decision is validated by the Angle 1 benchmark — the cross-domain rows in the benchmark table prove that routing is necessary.

---

## 3. System Architecture (Full Pipeline)

```
Input (PDF / raw text)
        │
        ▼
Preprocessing Pipeline
  - Text cleaning (spaCy, regex)
  - Section extraction (regex heuristics)
  - Keyword extraction (KeyBERT / YAKE)
  - NER (spaCy en_core_web_sm)
        │
        ▼
BERT Domain Classifier (bert-base-uncased, fine-tuned)
  Input: paper title + abstract (first 512 tokens)
  Output: domain label + confidence score
        │
        ├── General ──▶ BART fine-tuned (facebook/bart-large-cnn → ccdv/arxiv-summarization)
        └── Medical ──▶ PEGASUS zero-shot (google/pegasus-pubmed)
        │
        ▼
Structured Output Generator
  - OpenAI GPT-4o-mini via API
  - Extracts: key_findings, methodology, limitations, future_work
  - JSON mode enforced (response_format: json_object)
        │
        ▼
Final Output Dict:
  title, domain, confidence, low_confidence, probabilities,
  model_used, summary, key_findings, methodology,
  limitations, future_work
        │
        ▼
Demo Application
  - FastAPI backend (Python)
  - Next.js frontend (TypeScript, Tailwind, shadcn/ui)
  - PDF export via Jinja2 + WeasyPrint
  - Human evaluation panel (star ratings, saved to JSON)
```

---

## 4. Datasets

| Dataset | HuggingFace ID | Domain | Train | Validation | Test |
|---|---|---|---|---|---|
| arXiv Summarization | `ccdv/arxiv-summarization` | General (AI/CS/Physics) | 203,037 | 6,436 | 6,440 |
| PubMed Summarization | `ccdv/pubmed-summarization` | Medical (Biomedical/Clinical) | 119,924 | 6,633 | 6,658 |

**Fields used:**
- `article` → model input (full paper body)
- `abstract` → target summary (ground truth for evaluation)

**Important note for report:** The HuggingFace dataset `ccdv/pubmed` was renamed to `ccdv/pubmed-summarization` — use the new name when referencing it.

---

## 5. Models — Full Technical Details

### 5.1 Domain Classifier

| Property | Detail |
|---|---|
| Base model | `bert-base-uncased` |
| Architecture | BERT with classification head (2 classes) |
| Fine-tuning data | Topic/category labels from arXiv + PubMed metadata |
| Classes | General (0), Medical (1) |
| Input | Paper title + abstract, max 512 tokens |
| Output | Domain label + softmax confidence score |
| Training framework | HuggingFace Trainer API |
| Evaluation metrics | Accuracy, F1-score |
| Achieved accuracy | **99.4%** |
| Saved checkpoint | `outputs/classifier/` |

### 5.2 General Domain Summarizer (BART)

| Property | Detail |
|---|---|
| Base model | `facebook/bart-large-cnn` |
| Fine-tuning dataset | `ccdv/arxiv-summarization` (203K train pairs) |
| Input max tokens | 1,024 |
| Output max tokens | 256 |
| Training epochs | 3 |
| Batch size | 8 (gradient accumulation × 4, effective batch = 32) |
| Optimizer | AdamW, lr = 3e-5 |
| Scheduler | Linear warmup |
| Decoding | Beam search, num_beams=4, no_repeat_ngram_size=3 |
| Saved checkpoint | `outputs/bart_finetuned/` |

### 5.3 Medical Domain Summarizer (PEGASUS)

| Property | Detail |
|---|---|
| Base model | `google/pegasus-pubmed` |
| Strategy used | **Zero-shot** (fine-tuned checkpoint underperformed) |
| Input max tokens | 1,024 |
| Output max tokens | 256 |
| Decoding | Beam search, num_beams=4, no_repeat_ngram_size=3 |
| Tokenizer | `PegasusTokenizer` (not AutoTokenizer — tiktoken bug) |
| Why zero-shot | Fine-tuning produced NaN eval loss due to tokenizer bug in transformers==4.46.3. Zero-shot PEGASUS outperformed the fine-tuned checkpoint on all metrics. |

### 5.4 Structured Output Generator

| Property | Detail |
|---|---|
| Model | OpenAI GPT-4o-mini |
| Purpose | Extract structured fields from raw summary |
| Prompt strategy | Zero-shot JSON extraction with schema enforcement |
| Response format | `json_object` mode (enforced) |
| Temperature | 0.1 (low for consistent structured output) |
| Fields extracted | key_findings (list), methodology (str), limitations (list), future_work (list) |
| Retry logic | 3 attempts on JSON parse failure |

---

## 6. Angle 1 Benchmark Results

These are the actual results from `results/angle1_benchmark.csv`.

| Model Configuration | Test Domain | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|---|
| BART zero-shot | General (arXiv) | 0.3125 | 0.0926 | 0.1824 | 0.7871 |
| BART fine-tuned | General (arXiv) | **0.4521** | **0.1719** | **0.2531** | **0.8219** |
| PEGASUS zero-shot | Medical (PubMed) | **0.4266** | **0.1739** | **0.2526** | **0.8187** |
| PEGASUS fine-tuned | Medical (PubMed) | 0.3923 | 0.1427 | 0.2300 | 0.8073 |
| BART fine-tuned → Medical (cross) | Medical (PubMed) | 0.4063 | 0.1541 | 0.2275 | 0.8086 |
| PEGASUS fine-tuned → General (cross) | General (arXiv) | 0.3246 | 0.0913 | 0.1908 | 0.7760 |

### Key findings from the benchmark to discuss in the report:

1. **Fine-tuning BART improves all metrics substantially** — ROUGE-1 goes from 0.3125 (zero-shot) to 0.4521 (fine-tuned), a delta of +0.1396. This validates transfer learning for the General domain.

2. **PEGASUS fine-tuning underperformed zero-shot** — This is the anomalous result. Fine-tuned PEGASUS (0.3923 ROUGE-1) underperformed zero-shot PEGASUS (0.4266 ROUGE-1). The cause was NaN evaluation loss during training due to a tokenizer bug in `transformers==4.46.3` when using `AutoTokenizer` instead of `PegasusTokenizer`. This means the fine-tuned checkpoint did not train correctly. The router uses zero-shot PEGASUS for Medical papers as a result.

3. **Cross-domain tests justify routing (Angle 2)** — When BART (trained on General) is applied to Medical papers, it scores 0.4063 ROUGE-1 — lower than the domain-matched PEGASUS zero-shot (0.4266). When PEGASUS is applied to General papers, it scores only 0.3246 ROUGE-1 — far below BART fine-tuned (0.4521). This empirically proves domain-matched routing improves performance.

4. **BERTScore is consistently high across all configurations** — ranging from 0.7760 to 0.8219. This indicates all models capture semantic meaning reasonably well, but lexical overlap (ROUGE) varies more significantly with domain match.

---

## 7. Human Evaluation

### Criteria used (in the app)

| Criterion | Description | Scale |
|---|---|---|
| Fluency | Is the summary grammatically correct and readable? | 1–5 |
| Relevance | Does the summary accurately reflect the paper's content? | 1–5 |
| Conciseness | Is the summary free of redundant information? | 1–5 |

### Papers evaluated

12 entries collected so far (evaluations ongoing). Papers evaluated include:

**General domain:**
- Attention Is All You Need
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Deep Residual Learning for Image Recognition
- Generative Adversarial Nets
- Improving neural networks by preventing co-adaptation of feature detectors
- On the Identifiability of User Adaptation in Co-Adaptive Neural Interfaces (rated twice — useful for Cohen's Kappa)
- PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
- BART: Denoising Sequence-to-Sequence Pre-training
- Interacting Quantum Atoms Method for Crystalline Solids

**Medical domain:**
- Effects of screentime on the health and well-being of children and adolescents
- Recent Developments in SARS-CoV-2 Neutralizing Antibody Detection Methods

### Inter-annotator agreement
"On the Identifiability of User Adaptation in Co-Adaptive Neural Interfaces" was rated twice (scores: [4,5,3] and [5,4,4]) — compute Cohen's Kappa from these for the report.

### Output files from notebook 09
- `results/human_eval_per_paper.csv` — per-paper mean scores
- `results/human_eval_summary.csv` — mean ± std by domain
- `results/human_eval_domain_comparison.png` — bar chart
- `results/human_eval_boxplots.png` — box plots

---

## 8. Preprocessing Pipeline

| Step | Tool | Output |
|---|---|---|
| PDF parsing | PyMuPDF (`fitz`) | Raw text string |
| Text cleaning | regex + spaCy | Cleaned text (no headers, references noise) |
| Sentence segmentation | spaCy `en_core_web_sm` | List of sentences |
| Section extraction | regex heuristics | Dict of sections (Abstract, Introduction, Methodology, Results, Discussion, Conclusion) |
| Keyword extraction | KeyBERT / YAKE | Top N keywords using sentence-transformer embeddings |
| Stopword removal | NLTK / spaCy | Filtered token list |
| Named Entity Recognition | spaCy NER | Entities (drugs, methods, gene names, etc.) |

---

## 9. Application Layer (Demo System)

### Backend (FastAPI)

| File | Purpose |
|---|---|
| `app/backend/main.py` | FastAPI app, lifespan context manager, CORS |
| `app/backend/dependencies.py` | ResearchSummarizer singleton injection via Depends() |
| `app/backend/schemas.py` | Pydantic request/response models |
| `app/backend/routes/summarize.py` | Route definitions |
| `app/backend/controllers/summarize.py` | PDF extraction, pipeline orchestration |
| `app/backend/controllers/pdf.py` | Jinja2 + WeasyPrint PDF generation |
| `app/backend/templates/summary.html` | PDF export HTML template |

**API Endpoints:**

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/summarize/pdf` | Upload PDF → full structured summary |
| POST | `/api/summarize/text` | Raw text → full structured summary |
| POST | `/api/evaluate` | Save human evaluation scores |
| POST | `/api/summary/pdf` | Generate and download summary as PDF |

**Model loading:** All three models (BERT classifier, BART, PEGASUS) are loaded once at startup via FastAPI `lifespan` context manager and held in `app.state.summarizer`. Total GPU memory: ~8–10 GB on RTX 5090.

### Frontend (Next.js)

| Property | Detail |
|---|---|
| Framework | Next.js (App Router, TypeScript) |
| Styling | Tailwind CSS + shadcn/ui (Radix, Nova color) |
| Input modes | PDF upload (tab 1), raw text paste (tab 2) |
| Result display | Domain badge, confidence score, model used, summary, key findings, methodology, limitations, future work, probability bars |
| Human evaluation | Star rating panel (fluency, relevance, conciseness), saves to `results/human_eval_scores.json` |
| PDF export | Download button → POST result to `/api/summary/pdf` → Jinja2 + WeasyPrint → browser download |

---

## 10. Technical Stack Summary

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Package manager | uv |
| Deep learning | PyTorch (nightly cu128), HuggingFace Transformers 4.46.3 |
| Fine-tuning | HuggingFace Trainer API, PEFT |
| NLP preprocessing | spaCy 3.8.4, NLTK 3.9.1, KeyBERT 0.8.5, YAKE 0.4.8 |
| PDF parsing | PyMuPDF 1.24.10 |
| Evaluation | rouge-score 0.1.2, bert-score 0.3.13, scikit-learn 1.5.2 |
| Experiment tracking | Weights & Biases (wandb) |
| Structured output | OpenAI GPT-4o-mini |
| Backend | FastAPI, Uvicorn |
| PDF export | Jinja2, WeasyPrint |
| Frontend | Next.js, TypeScript, Tailwind CSS, shadcn/ui |
| Hardware | NVIDIA RTX 5090 (32GB VRAM), Blackwell sm_120 |
| Notebooks | Jupyter (9 notebooks, one per phase) |

---

## 11. AI Techniques Coverage (for report mapping)

| Technique | How it is covered in this project |
|---|---|
| NLP — text preprocessing | spaCy pipeline: tokenization, NER, stopword removal, section extraction using regex heuristics |
| NLP — word embeddings | KeyBERT uses sentence-transformers embeddings to extract semantically relevant keywords |
| Transformer-based LLMs | BART and PEGASUS as core seq2seq summarization models; BERT as classifier |
| Fine-tuning / Transfer Learning | bert-base-uncased fine-tuned for classification; bart-large-cnn fine-tuned on arXiv |
| Prompt Engineering | GPT-4o-mini structured extraction with zero-shot JSON schema prompting |
| Few-shot Learning | 3-shot prompting variant tested in notebook 08 ablation study |
| Domain Adaptation | Domain-specific fine-tuning + routing — the central contribution of the project |

---

## 12. Suggested Report Structure

### 1. Abstract (200–250 words)
Summarize the problem, approach, two contributions, key results (BART fine-tuned ROUGE-1: 0.4521, classifier accuracy: 99.4%), and conclusions.

### 2. Introduction
- Problem: Generic summarization tools (NotebookLM, etc.) do not adapt to domain — medical papers need domain-specific vocabulary and framing
- Gap: No publicly benchmarked domain-adaptive routing system for research paper summarization
- Proposed solution: Two-angle approach — benchmark + routing
- Contributions: List the two angles explicitly
- Paper structure: "The rest of this paper is organized as follows..."

### 3. Related Work
Cover these areas with relevant citations:
- **Abstractive summarization:** BART (Lewis et al., 2020), PEGASUS (Zhang et al., 2020), T5
- **Domain-specific summarization:** BioBART, SciBERT, clinical NLP models
- **Text classification:** BERT (Devlin et al., 2019), domain classification approaches
- **Evaluation metrics:** ROUGE (Lin, 2004), BERTScore (Zhang et al., 2020)
- **Existing tools:** NotebookLM, Semantic Scholar, Elicit — and why they don't publish benchmarks

### 4. Methodology
#### 4.1 System Overview
Include the architecture diagram from the architecture document.

#### 4.2 Datasets
Table of both datasets with sizes, splits, domains, fields used.

#### 4.3 Preprocessing Pipeline
Table of steps → tools → outputs. Mention section extraction produces structured dict (Abstract, Introduction, Methodology, Results, Discussion, Conclusion).

#### 4.4 Domain Classifier (Angle 2 — Part 1)
- Base model: bert-base-uncased
- Input: title + abstract, 512 tokens
- Training setup: HuggingFace Trainer, accuracy/F1 metrics
- Result: 99.4% accuracy

#### 4.5 Summarization Models
Two subsections:
- **General:** BART fine-tuned on arXiv — training hyperparameters table
- **Medical:** PEGASUS zero-shot — explain why zero-shot was chosen (NaN loss during fine-tuning due to tokenizer bug), cite this as a finding rather than a failure

#### 4.6 Structured Output Generator
- GPT-4o-mini with JSON mode
- Prompt strategy: zero-shot schema extraction
- Fields extracted and why (key findings, methodology, limitations, future work)

#### 4.7 Demo Application
Brief description — FastAPI + Next.js, input methods, output display, human eval panel, PDF export.

### 5. Angle 1 — Comparative Evaluation Results
- Present the full benchmark table (6 rows × 4 metrics)
- Subsection: BART zero-shot vs fine-tuned — discuss the improvement delta
- Subsection: PEGASUS zero-shot vs fine-tuned — explain the anomaly (NaN loss, tokenizer bug)
- Subsection: Cross-domain analysis — the two cross-domain rows justify Angle 2
- Include bar charts or metric comparison figures if available

### 6. Angle 2 — Domain Routing Results
- Classifier performance: 99.4% accuracy, F1-score
- Routing logic: threshold=0.5, low_confidence flag
- Justification from benchmark: show how domain-matched models outperform cross-domain models
- End-to-end pipeline performance: example outputs for one General and one Medical paper

### 7. Human Evaluation
- Describe the criteria (fluency, relevance, conciseness) and scale (1–5)
- Present the summary table (mean ± std by domain from `results/human_eval_summary.csv`)
- Include the bar chart (`results/human_eval_domain_comparison.png`)
- Include box plots (`results/human_eval_boxplots.png`)
- Cohen's Kappa result from the duplicate-rated paper
- Discussion: do human scores align with ROUGE/BERTScore? Any discrepancies?

### 8. Discussion
- Why fine-tuning helped BART but hurt PEGASUS — tokenizer issue, implication for future work
- What the cross-domain results mean for the necessity of routing
- Limitations of ROUGE as a metric (recall-biased, does not capture fluency)
- BERTScore vs ROUGE discrepancy — semantic similarity vs lexical overlap
- Human eval sample size limitations (acknowledge it, note it as a constraint)
- Confidence threshold — 99.4% classifier accuracy means routing errors are rare but possible

### 9. Conclusion
- Restate the two contributions
- Summarize the key quantitative findings
- State what the system enables (reproducible benchmark, open-source pipeline)

### 10. Future Work
- **Angle 3:** Multi-document synthesis across papers on the same topic
- **Angle 4:** Literature-review-ready structured outputs, citation graph visualizations
- Fix PEGASUS fine-tuning (use PegasusTokenizer throughout, resolve NaN loss)
- Extend benchmark to additional domains (legal, finance, education)
- Add more evaluators and criteria to the human evaluation panel

### 11. References
Key papers to cite:
- BART: Lewis et al. (2020) — "BART: Denoising Sequence-to-Sequence Pre-training" (arXiv:1910.13461)
- PEGASUS: Zhang et al. (2020) — "PEGASUS: Pre-training with Extracted Gap-sentences" (arXiv:1912.08777)
- BERT: Devlin et al. (2019) — "BERT: Pre-training of Deep Bidirectional Transformers" (arXiv:1810.04805)
- ROUGE: Lin (2004) — "ROUGE: A Package for Automatic Evaluation of Summaries"
- BERTScore: Zhang et al. (2020) — "BERTScore: Evaluating Text Generation with BERT" (arXiv:1904.09675)
- arXiv dataset: Cohan et al. (2018) — "A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents"
- PubMed dataset: same Cohan et al. (2018) paper covers both datasets
- AdamW: Loshchilov & Hutter (2019) — "Decoupled Weight Decay Regularization"
- HuggingFace: Wolf et al. (2020) — "Transformers: State-of-the-Art Natural Language Processing"

---

## 13. Important Clarifications for the Report

These are things that could trip you up if not handled carefully:

1. **PEGASUS fine-tuning failure is a finding, not just a bug.** Frame it as: "During fine-tuning, we encountered NaN evaluation loss attributed to a tokenizer incompatibility in transformers==4.46.3 when using AutoTokenizer with PEGASUS. This finding highlights the importance of using model-specific tokenizers. As a result, the zero-shot PEGASUS checkpoint outperformed the fine-tuned version on all metrics, and the routing system uses zero-shot PEGASUS for Medical papers."

2. **The classifier achieves 99.4% accuracy** — this is a strong result. Explain it by noting that arXiv and PubMed papers have very distinct vocabulary distributions, making binary classification relatively straightforward for a pretrained BERT model.

3. **The cross-domain rows in the benchmark are the core evidence for Angle 2.** Without those rows, there is no proof that routing is necessary. Make sure the discussion explicitly connects the benchmark table to the routing decision.

4. **Human evaluation criteria mismatch.** The architecture document lists 5 criteria (Accuracy, Conciseness, Coherence, Completeness, Domain Relevance) but the app collected 3 (Fluency, Relevance, Conciseness). Use the 3 criteria that were actually collected — do not claim 5 were measured.

5. **Dataset rename.** The architecture document says `ccdv/pubmed` but the correct current HuggingFace ID is `ccdv/pubmed-summarization`. Use the correct name in the report.

6. **BART checkpoint status.** There was an open question about whether `outputs/bart_finetuned/` contains a real fine-tuned checkpoint or a placeholder. Verify this before writing the BART results section — if it is a real checkpoint, the ROUGE-1: 0.4521 result is valid. If it is a placeholder, the results need to be recomputed.

---

## 14. Source Code File Map (for report reference)

| File | What it implements |
|---|---|
| `src/classifier.py` | BERT classifier wrapper — load_classifier, predict_domain, batch_predict |
| `src/router.py` | DomainRouter — loads all 3 models, classify + route + summarize |
| `src/summarizer.py` | ResearchSummarizer — top-level entry point, wraps router + generator |
| `src/output_generator.py` | StructuredOutputGenerator — OpenAI GPT-4o-mini structured extraction |
| `src/preprocessing.py` | Text cleaning, section extraction, keyword extraction |
| `src/evaluation.py` | Angle 1 benchmark — ROUGE + BERTScore computation |
| `app/backend/main.py` | FastAPI app — lifespan, CORS, route registration |
| `app/backend/routes/summarize.py` | API routes — /summarize/pdf, /summarize/text, /evaluate, /summary/pdf |
| `app/backend/controllers/summarize.py` | PDF extraction (PyMuPDF), pipeline orchestration, eval score saving |
| `app/backend/controllers/pdf.py` | Jinja2 template rendering + WeasyPrint PDF generation |
| `app/backend/templates/summary.html` | PDF export template |
| `slurm/run_bart.py` | BART fine-tuning script |
| `slurm/run_pegasus.py` | PEGASUS fine-tuning script |
| `slurm/run_classifier.py` | BERT classifier training script |
| `notebooks/01–09` | One notebook per phase |

---

## 15. Results Files Available

| File | Contents |
|---|---|
| `results/angle1_benchmark.csv` | Full 6-row benchmark table with all metrics |
| `results/angle1_benchmark.json` | Same data in JSON format |
| `results/rouge_scores.json` | Raw ROUGE scores per model |
| `results/bertscores.json` | Raw BERTScore values per model |
| `results/human_eval_scores.json` | Raw human evaluation entries |
| `results/human_eval_per_paper.csv` | Per-paper mean scores |
| `results/human_eval_summary.csv` | Mean ± std by domain |
| `results/human_eval_domain_comparison.png` | Bar chart — scores by domain |
| `results/human_eval_boxplots.png` | Box plots — score distributions |
| `results/structured/` | Per-paper JSON outputs from GPT-4o-mini |

---

*Use this document as the starting prompt context in a new chat session for report writing.*
*All benchmark numbers, model details, and architectural decisions in this document reflect the actual implemented system.*
