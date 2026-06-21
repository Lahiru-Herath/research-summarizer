#!/bin/bash
#SBATCH --job-name=led-arxiv
#SBATCH --output=outputs/led_logs/slurm_%j.out
#SBATCH --error=outputs/led_logs/slurm_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

mkdir -p outputs/led_logs outputs/led_finetuned results

# ── Run training ──────────────────────────────────────────────────────────────
python - <<'PYEOF'
import json, os, random
import numpy as np
import torch
import yaml
import wandb
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from torch.utils.data import Dataset as TorchDataset

# ── Config ────────────────────────────────────────────────────────────────────
with open("configs/led_config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED        = cfg["training"]["seed"]
MODEL_NAME  = cfg["model"]["name"]
MAX_INPUT   = cfg["model"]["max_input_tokens"]
MAX_OUTPUT  = cfg["model"]["max_output_tokens"]
OUTPUT_DIR  = cfg["output"]["output_dir"]
LOG_DIR     = cfg["output"]["logging_dir"]

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading ccdv/arxiv-summarization ...")
ds = load_dataset(cfg["data"]["dataset_id"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    """Tokenize article + abstract for LED seq2seq training.

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
    # LED requires global attention on the first token
    global_attention = [[1] + [0] * (MAX_INPUT - 1)] * len(batch[cfg["data"]["input_field"]])
    inputs["global_attention_mask"] = global_attention

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            batch[cfg["data"]["target_field"]],
            max_length=MAX_OUTPUT,
            truncation=True,
            padding="max_length",
        )
    # Replace padding token id with -100 so loss ignores it
    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in targets["input_ids"]
    ]
    inputs["labels"] = labels
    return inputs

print("Tokenizing dataset (this may take a while for 203K examples)...")
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
    fp16                        = True,
    evaluation_strategy         = t["evaluation_strategy"],
    save_strategy               = t["save_strategy"],
    load_best_model_at_end      = t["load_best_model_at_end"],
    metric_for_best_model       = t["metric_for_best_model"],
    predict_with_generate       = False,   # skip generation during training for speed
    seed                        = SEED,
    report_to                   = "wandb",
    run_name                    = cfg["wandb"]["run_name"],
    logging_steps               = 100,
    save_total_limit            = 2,       # keep only the 2 best checkpoints
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
PYEOF

echo "============================================"
echo "Finished : $(date)"
echo "============================================"
