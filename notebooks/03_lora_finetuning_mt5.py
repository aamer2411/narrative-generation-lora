"""
Notebook 03: LoRA Fine-Tuning - mT5-base (Objective 2 - Part B)
-----------------------------------------------------------------
Fine-tunes mT5-base on the Urdu prompt-continuation dataset using LoRA
(Low-Rank Adaptation) via the PEFT library.

Prerequisites:
    - Run data/prepare_dataset.py to generate urdu_train.csv and urdu_val.csv

Expected improvement (from thesis):
    ROUGE-L: 0.505 -> 0.541  (+7.1%)
    BERTScore: 0.887 -> 0.898

LoRA config: r=16, alpha=32, target_modules=["q","v"], dropout=0.05

Environment: Google Colab T4 GPU (15 GB VRAM), FP16 mixed precision
"""

# =============================================================================
# 1. Install Dependencies (uncomment when running in Colab)
# =============================================================================

# !pip install -q transformers peft datasets evaluate rouge-score bert-score accelerate sentencepiece

# =============================================================================
# 2. Imports
# =============================================================================

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
from rouge_score import rouge_scorer
from bert_score import score as bert_score

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID   = "google/mt5-base"
OUTPUT_DIR = "./mt5_lora_urdu"
DATA_DIR   = "../data"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# 3. Load Tokenizer and Apply LoRA
# =============================================================================

print("\nLoading mT5-base tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(DEVICE)

# =============================================================================
# 4. Load and Preprocess Dataset
# =============================================================================

TRAIN_CSV = os.path.join(DATA_DIR, "urdu_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "urdu_val.csv")

assert os.path.exists(TRAIN_CSV), (
    f"Training data not found at {TRAIN_CSV}. "
    "Run data/prepare_dataset.py first."
)

print(f"\nLoading dataset from {DATA_DIR}...")
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
print(f"  Train: {len(train_df):,} pairs")
print(f"  Val:   {len(val_df):,} pairs")

MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 128


def preprocess(batch):
    model_inputs = tokenizer(
        ["summarize: " + p for p in batch["prompt"]],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=list(batch["continuation"]),
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )
    # Replace padding token id in labels with -100 so loss ignores them
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing datasets...")
train_dataset = Dataset.from_pandas(train_df[["prompt", "continuation"]])
val_dataset   = Dataset.from_pandas(val_df[["prompt",   "continuation"]])

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["prompt", "continuation"])
val_dataset   = val_dataset.map(preprocess,   batched=True, remove_columns=["prompt", "continuation"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# =============================================================================
# 5. Training Arguments
# =============================================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    save_total_limit=1,
)

# =============================================================================
# 6. Train
# =============================================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\nStarting LoRA fine-tuning of mT5-base...")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  FP16: {training_args.fp16}")

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")

# =============================================================================
# 7. Post-Training Evaluation on XL-Sum Test Set
# =============================================================================

from datasets import load_dataset as hf_load

print("\nEvaluating on XL-Sum Urdu test set...")
xlsum      = hf_load("GEM/xlsum", "urdu", split="test", trust_remote_code=True)
test_sample = xlsum.select(range(100))

model.eval()


def mt5_summarize(text: str) -> str:
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TARGET_LEN, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


predictions = [mt5_summarize(row["document"][:512]) for row in test_sample]
references  = [row["target"] for row in test_sample]

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
rougel_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
rougel_avg    = sum(rougel_scores) / len(rougel_scores)

_, _, f1 = bert_score(predictions, references, lang="ur", verbose=False)
bert_avg = f1.mean().item()

print("\n" + "="*45)
print("mT5 LoRA Fine-Tuned - Evaluation Results")
print("="*45)
print(f"  ROUGE-L:   {rougel_avg:.3f}  (thesis target: 0.541)")
print(f"  BERTScore: {bert_avg:.3f}  (thesis target: 0.898)")
print(f"\n  Improvement in ROUGE-L over baseline (0.505): "
      f"{((rougel_avg - 0.505) / 0.505 * 100):+.1f}%")
print(f"  (Thesis reports +7.1% improvement)")
