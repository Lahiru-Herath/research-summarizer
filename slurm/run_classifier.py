"""
run_classifier.py
-----------------
Standalone training script for the BERT domain classifier.
Equivalent to slurm/train_classifier.sbatch but runs directly with:

    uv run python slurm/run_classifier.py
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Env & config ──────────────────────────────────────────────────────────────
load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ── Data ──────────────────────────────────────────────────────────────────────
samples = cfg["data"]["samples_per_class"]
print(f"Loading {samples} samples per class ...")

arxiv_ds  = load_dataset(cfg["data"]["arxiv_dataset_id"],  split="train").shuffle(seed=SEED).select(range(samples))
pubmed_ds = load_dataset(cfg["data"]["pubmed_dataset_id"], split="train").shuffle(seed=SEED).select(range(samples))

def to_cls(ds, label_id):
    """Convert a HuggingFace dataset split into (text, label) records.

    Args:
        ds:       HuggingFace Dataset split.
        label_id: Integer class label (0=General, 1=Medical).

    Returns:
        List of dicts with keys "text" and "label".
    """
    recs = []
    for row in ds:
        title    = (row.get("title")    or "").strip()
        abstract = (row.get("abstract") or "").strip()
        text     = (title + " [SEP] " + abstract).strip()
        recs.append({"text": text, "label": label_id})
    return recs

all_records = to_cls(arxiv_ds, 0) + to_cls(pubmed_ds, 1)
random.shuffle(all_records)

n       = len(all_records)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train_r = all_records[:n_train]
val_r   = all_records[n_train:n_train + n_val]
test_r  = all_records[n_train + n_val:]

print(f"Splits — train: {len(train_r)}, val: {len(val_r)}, test: {len(test_r)}")

hf_ds = DatasetDict({
    "train":      Dataset.from_list(train_r),
    "validation": Dataset.from_list(val_r),
    "test":       Dataset.from_list(test_r),
})

# ── Tokenize ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    """Tokenize a batch of classifier input texts.

    Args:
        batch: Dict with key "text" (list of strings).

    Returns:
        Dict of input_ids, attention_mask, token_type_ids.
    """
    return tokenizer(
        batch["text"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )

tokenized = hf_ds.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Compute accuracy and weighted F1 from Trainer eval predictions.

    Args:
        eval_pred: EvalPrediction namedtuple (predictions, label_ids).

    Returns:
        Dict with keys "accuracy" and "f1".
    """
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
    bf16                        = True,
    eval_strategy               = t["eval_strategy"],
    save_strategy               = t["save_strategy"],
    load_best_model_at_end      = t["load_best_model_at_end"],
    metric_for_best_model       = t["metric_for_best_model"],
    greater_is_better           = True,
    seed                        = SEED,
    report_to                   = "wandb",
    run_name                    = cfg["wandb"]["run_name"],
    logging_steps               = 50,
    save_total_limit            = 2,
    dataloader_num_workers      = 4,
    dataloader_pin_memory       = True,
)

wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

trainer = Trainer(
    model            = model,
    args             = args,
    train_dataset    = tokenized["train"],
    eval_dataset     = tokenized["validation"],
    processing_class = tokenizer,
    compute_metrics  = compute_metrics,
    callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training classifier ...")
trainer.train()

# ── Test evaluation ───────────────────────────────────────────────────────────
print("Evaluating on test set ...")
out    = trainer.predict(tokenized["test"])
preds  = np.argmax(out.predictions, axis=-1)
labels = out.label_ids

acc = accuracy_score(labels, preds)
f1  = f1_score(labels, preds, average="weighted")
cm  = confusion_matrix(labels, preds).tolist()

print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1       : {f1:.4f}")
print(f"Confusion matrix:\n{np.array(cm)}")

Path("results").mkdir(exist_ok=True)
with open("results/classifier_test_metrics.json", "w", encoding="utf-8") as fout:
    json.dump({"accuracy": acc, "f1": f1, "confusion_matrix": cm}, fout, indent=2)
print("Saved test metrics → results/classifier_test_metrics.json")

# ── Save ──────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved → {OUTPUT_DIR}")

wandb.finish()