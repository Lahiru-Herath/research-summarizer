"""
run_led.py
----------
Standalone training script for LED fine-tuning on arXiv.
Equivalent to slurm/train_led_arxiv.sbatch but runs directly with:

    uv run python slurm/run_led.py
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

with open("configs/led_config.yaml") as f:
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    """Tokenize article + abstract for LED seq2seq training.

    LED requires a global_attention_mask with 1 on the first token
    so the model attends globally from the start token.

    Args:
        batch: Dict with keys matching cfg data fields.

    Returns:
        Dict of input_ids, attention_mask, global_attention_mask, labels.
    """
    inputs = tokenizer(
        batch[cfg["data"]["input_field"]],
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
    )
    # LED-specific: set global attention on the first token of every example
    global_attention = [
        [1] + [0] * (MAX_INPUT - 1)
        for _ in batch[cfg["data"]["input_field"]]
    ]
    inputs["global_attention_mask"] = global_attention

    targets = tokenizer(
    batch[cfg["data"]["target_field"]],
    max_length=MAX_OUTPUT,
    truncation=True,
    padding="max_length",
)
    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in targets["input_ids"]
    ]
    inputs["labels"] = labels
    return inputs

print("Tokenizing dataset (203K examples at 16K tokens — this takes a while) ...")
tokenized = ds.map(
    tokenize,
    batched=True,
    batch_size=8,          # small batch size — LED inputs are very long
    remove_columns=ds["train"].column_names,
)
tokenized.set_format("torch")
print(f"Train: {len(tokenized['train']):,}  Val: {len(tokenized['validation']):,}")

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()
model.config.use_cache = False      # required when gradient checkpointing is on

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
    fp16                        = False,
    bf16                        = True,
    evaluation_strategy         = t["evaluation_strategy"],
    save_strategy               = t["save_strategy"],
    load_best_model_at_end      = t["load_best_model_at_end"],
    metric_for_best_model       = t["metric_for_best_model"],
    predict_with_generate       = False,
    seed                        = SEED,
    report_to                   = "wandb",
    run_name                    = cfg["wandb"]["run_name"],
    logging_steps               = 100,
    save_total_limit            = 2,
)

wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model         = model,
    args          = args,
    train_dataset = tokenized["train"],
    eval_dataset  = tokenized["validation"],
    tokenizer     = tokenizer,
    data_collator = data_collator,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training LED on arXiv ...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved → {OUTPUT_DIR}")

wandb.finish()
