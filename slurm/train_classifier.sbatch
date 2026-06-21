#!/bin/bash
#SBATCH --job-name=bert-classifier
#SBATCH --output=outputs/classifier_logs/slurm_%j.out
#SBATCH --error=outputs/classifier_logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# CLUSTER-SPECIFIC: uncomment and set your partition
# #SBATCH --partition=<YOUR_PARTITION>
# CLUSTER-SPECIFIC: uncomment and set your account if required
# #SBATCH --account=<YOUR_ACCOUNT>

set -euo pipefail

echo "============================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "============================================"

# CLUSTER-SPECIFIC: load your HPC modules, e.g.:
# module load cuda/12.1
# module load python/3.11

# Load .env (WANDB_API_KEY, HF_DATASETS_CACHE, HF_HOME)
if [ -f ".env" ]; then
    set -a && source .env && set +a
    echo ".env sourced"
fi

# Activate uv virtualenv
source .venv/bin/activate
echo "Python  : $(python --version)"
echo "CUDA    : $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

mkdir -p outputs/classifier_logs outputs/classifier results

# ── Run training ──────────────────────────────────────────────────────────────
python - <<'PYEOF'
import json, os, random
import numpy as np
import torch
import yaml
import wandb
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
with open("configs/classifier_config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED       = cfg["training"]["seed"]
MODEL_NAME = cfg["model"]["name"]
MAX_LEN    = cfg["model"]["max_input_tokens"]
NUM_LABELS = cfg["model"]["num_labels"]
LABEL_MAP  = {int(k): v for k, v in cfg["model"]["label_map"].items()}
OUTPUT_DIR = cfg["output"]["output_dir"]
LOG_DIR    = cfg["output"]["logging_dir"]

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
samples = cfg["data"]["samples_per_class"]
print(f"Loading {samples} samples per class…")

arxiv_ds  = load_dataset("ccdv/arxiv-summarization", split="train").shuffle(seed=SEED).select(range(samples))
pubmed_ds = load_dataset("ccdv/pubmed",              split="train").shuffle(seed=SEED).select(range(samples))

def to_cls(ds, label_id):
    recs = []
    for row in ds:
        title    = (row.get("title")    or "").strip()
        abstract = (row.get("abstract") or "").strip()
        recs.append({"text": title + " [SEP] " + abstract, "label": label_id})
    return recs

all_records = to_cls(arxiv_ds, 0) + to_cls(pubmed_ds, 1)
random.shuffle(all_records)

n         = len(all_records)
n_train   = int(0.8 * n)
n_val     = int(0.1 * n)
train_r   = all_records[:n_train]
val_r     = all_records[n_train:n_train + n_val]
test_r    = all_records[n_train + n_val:]

print(f"Splits — train: {len(train_r)}, val: {len(val_r)}, test: {len(test_r)}")

hf_ds = DatasetDict({
    "train":      Dataset.from_list(train_r),
    "validation": Dataset.from_list(val_r),
    "test":       Dataset.from_list(test_r),
})

# ── Tokenize ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], max_length=MAX_LEN, truncation=True, padding="max_length")

tokenized = hf_ds.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# ── Model ─────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

# ── Training args ─────────────────────────────────────────────────────────────
t = cfg["training"]
args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    logging_dir                 = LOG_DIR,
    num_train_epochs            = t["epochs"],
    per_device_train_batch_size = t["per_device_train_batch_size"],
    per_device_eval_batch_size  = t["per_device_eval_batch_size"],
    gradient_accumulation_steps = t["gradient_accumulation_steps"],
    optim                       = t["optimizer"],
    learning_rate               = t["learning_rate"],
    lr_scheduler_type           = t["lr_scheduler_type"],
    warmup_ratio                = t["warmup_ratio"],
    weight_decay                = t["weight_decay"],
    fp16                        = True,
    evaluation_strategy         = t["evaluation_strategy"],
    save_strategy               = t["save_strategy"],
    load_best_model_at_end      = t["load_best_model_at_end"],
    metric_for_best_model       = t["metric_for_best_model"],
    greater_is_better           = True,
    seed                        = SEED,
    report_to                   = "wandb",
    run_name                    = cfg["wandb"]["run_name"],
    logging_steps               = 50,
)

wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = tokenized["train"],
    eval_dataset    = tokenized["validation"],
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Training…")
trainer.train()

# ── Test evaluation ───────────────────────────────────────────────────────────
out    = trainer.predict(tokenized["test"])
preds  = np.argmax(out.predictions, axis=-1)
labels = out.label_ids

acc = accuracy_score(labels, preds)
f1  = f1_score(labels, preds, average="weighted")
cm  = confusion_matrix(labels, preds).tolist()

print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1       : {f1:.4f}")
print(f"Confusion matrix: {cm}")

Path("results").mkdir(exist_ok=True)
with open("results/classifier_test_metrics.json", "w", encoding="utf-8") as fout:
    json.dump({"accuracy": acc, "f1": f1, "confusion_matrix": cm}, fout, indent=2)

# ── Save ──────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved → {OUTPUT_DIR}")

wandb.finish()
PYEOF

echo "============================================"
echo "Finished : $(date)"
echo "============================================"
