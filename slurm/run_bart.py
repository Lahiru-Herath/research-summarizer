"""
run_bart.py
-----------
Standalone training script for BART fine-tuning on arXiv.
Run directly with:

    uv run python slurm/run_bart.py
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# ── Env & config ──────────────────────────────────────────────────────────────
load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

with open("configs/bart_config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED       = cfg["training"]["seed"]
MODEL_NAME = cfg["model"]["name"]
MAX_INPUT  = cfg["model"]["max_input_tokens"]
MAX_OUTPUT = cfg["model"]["max_output_tokens"]
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
print(f"Loading {cfg['data']['dataset_id']} ...")
ds = load_dataset(cfg["data"]["dataset_id"])

print(f"Model name: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    """Tokenize article + abstract for BART seq2seq training.

    Args:
        batch: Dict with keys matching cfg data fields.

    Returns:
        Dict of input_ids, attention_mask, labels.
    """
    inputs = tokenizer(
        batch[cfg["data"]["input_field"]],
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
    )
    targets = tokenizer(
        batch[cfg["data"]["target_field"]],
        max_length=MAX_OUTPUT,
        truncation=True,
        padding="max_length",
    )
    labels = [
        np.array([(t if t != tokenizer.pad_token_id else -100) for t in label], dtype=np.int64)
        for label in targets["input_ids"]
    ]
    inputs["labels"] = labels
    return inputs

print("Tokenizing dataset ...")
tokenized = ds.map(
    tokenize,
    batched=True,
    batch_size=64,
    remove_columns=ds["train"].column_names,
)
tokenized.set_format("torch")
print(f"Train: {len(tokenized['train']):,}  Val: {len(tokenized['validation']):,}")

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ── Training args ─────────────────────────────────────────────────────────────
t = cfg["training"]
args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    logging_dir                 = LOG_DIR,
    num_train_epochs            = t["epochs"],
    per_device_train_batch_size = t["per_device_train_batch_size"],
    per_device_eval_batch_size  = t["per_device_eval_batch_size"],
    gradient_accumulation_steps = t["gradient_accumulation_steps"],
    optim                       = t["optimizer"],
    learning_rate               = t["learning_rate"],
    lr_scheduler_type           = t["lr_scheduler_type"],
    warmup_steps                = t["warmup_steps"],
    weight_decay                = t["weight_decay"],
    bf16                        = True,
    evaluation_strategy         = t["evaluation_strategy"],
    save_strategy               = t["save_strategy"],
    load_best_model_at_end      = t["load_best_model_at_end"],
    metric_for_best_model       = t["metric_for_best_model"],
    predict_with_generate       = False,
    seed                        = SEED,
    report_to                   = "wandb",
    run_name                    = cfg["wandb"]["run_name"],
    logging_steps               = 500,
    save_total_limit            = 2,
    dataloader_num_workers      = 4,
    dataloader_pin_memory       = True,
)

wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model            = model,
    args             = args,
    train_dataset    = tokenized["train"],
    eval_dataset     = tokenized["validation"],
    processing_class = tokenizer,       # replaces deprecated tokenizer= arg
    data_collator    = data_collator,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training BART on arXiv ...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved → {OUTPUT_DIR}")

wandb.finish()