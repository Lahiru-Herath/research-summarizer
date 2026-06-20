# Domain-Adaptive Multi-Model Research Paper Summarization System

An end-to-end AI pipeline that classifies research papers by domain and routes
them to the best-performing domain-specific summarization model, backed by a
rigorous empirical benchmark.

## Research Contributions

- **Angle 1 — Comparative Evaluation Framework:** ROUGE + BERTScore benchmark
  comparing pretrained vs fine-tuned models across general and medical domains,
  including cross-domain tests that justify the routing system.
- **Angle 2 — Domain-Aware Model Routing:** A fine-tuned BERT classifier that
  routes each paper to its optimal summarizer (LED for general, PEGASUS for medical).

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Install

```bash
# Clone the repo
git clone <your-repo-url>
cd research-summarizer

# Create virtual environment and activate
uv venv --python 3.11
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install all dependencies
uv add torch transformers accelerate datasets tokenizers peft \
    spacy nltk keybert yake pymupdf langchain langchain-community \
    rouge-score bert-score scikit-learn wandb sentence-transformers \
    jupyter ipykernel matplotlib seaborn pandas numpy tqdm \
    python-dotenv pyyaml

# Download spaCy model
uv run python -m spacy download en_core_web_sm

# Download NLTK data
uv run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Environment Variables

Create a `.env` file in the project root:

```
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_huggingface_token_here   # only needed for gated models
```

---

## Running the Notebooks

Run notebooks in order:

```bash
jupyter notebook notebooks/
```

| #   | Notebook                               | Purpose                                     |
| --- | -------------------------------------- | ------------------------------------------- |
| 01  | `01_data_loading_and_eda.ipynb`        | Load datasets, EDA, sanity checks           |
| 02  | `02_preprocessing_pipeline.ipynb`      | Text cleaning, section + keyword extraction |
| 03  | `03_domain_classifier.ipynb`           | BERT classifier fine-tuning                 |
| 04  | `04_led_finetuning.ipynb`              | LED fine-tuning on arXiv                    |
| 05  | `05_pegasus_finetuning.ipynb`          | PEGASUS fine-tuning on PubMed               |
| 06  | `06_evaluation_and_comparison.ipynb`   | Angle 1 — full benchmark table              |
| 07  | `07_routing_system.ipynb`              | Angle 2 — end-to-end routing demo           |
| 08  | `08_structured_output_prompting.ipynb` | Prompt engineering ablation                 |
| 09  | `09_human_evaluation.ipynb`            | Human rubric + Cohen's Kappa                |

---

## Submitting SLURM Jobs (HPC)

```bash
# Submit the two longest training jobs first, simultaneously
sbatch slurm/train_led_arxiv.sbatch
sbatch slurm/train_pegasus_pubmed.sbatch

# Submit classifier training (can start once data prep is done)
sbatch slurm/train_classifier.sbatch

# Submit evaluations after all training jobs complete
sbatch slurm/evaluate_all.sbatch
```

Check job status:

```bash
squeue -u $USER
```

---

## Reproducing Results

1. Complete setup above
2. Run notebooks 01 and 02 to prepare data
3. Submit SLURM training jobs as above
4. Once training is complete, run notebook 06 (`evaluate_all.sbatch` handles this on HPC)
5. Results are saved to `results/rouge_scores.json`, `results/bertscores.json`, `results/human_eval.csv`

---

## Project Structure

```
research-summarizer/
├── pyproject.toml
├── README.md
├── .gitignore
├── data/                  # HuggingFace datasets (not committed)
├── notebooks/             # One notebook per component
├── src/                   # Reusable Python modules
│   ├── preprocessing.py
│   ├── classifier.py
│   ├── summarizer.py
│   ├── router.py
│   ├── output_generator.py
│   └── evaluation.py
├── slurm/                 # SLURM job scripts for HPC
├── configs/               # Hyperparameter YAML files
├── outputs/               # Model checkpoints (not committed)
└── results/               # Evaluation results (committed)
```
