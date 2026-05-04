"""
Notebook 04: LoRA Fine-Tuning - GPT-2-Urdu (Objective 2 - Part B)
-------------------------------------------------------------------
Fine-tunes GPT-2-Urdu (Imran1/gpt2-urdu-news) for story continuation
using LoRA via the PEFT library.

Prerequisites:
    - Run data/prepare_dataset.py to generate urdu_train.csv and urdu_val.csv

Expected improvement (from thesis):
    ROUGE-L: 0.462 -> 0.507  (+9.7%)
    BERTScore: 0.873 -> 0.882

LoRA config: r=8, alpha=16, target_modules=["c_attn"], dropout=0.05

Environment: Google Colab T4 GPU (15 GB VRAM), FP16 mixed precision
"""

# =============================================================================
# 1. Install Dependencies (uncomment when running in Colab)
# =============================================================================

# !pip install -q transformers peft datasets evaluate rouge-score bert-score accelerate

# =============================================================================
# 2. Imports
# =============================================================================

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from rouge_score import rouge_scorer
from bert_score import score as bert_score

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID   = "Imran1/gpt2-urdu-news"
OUTPUT_DIR = "./gpt2_lora_urdu"
DATA_DIR   = "../data"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# 3. Load Tokenizer and Apply LoRA
# =============================================================================

print("\nLoading GPT-2-Urdu tokenizer and model...")
tokenizer            = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token  = tokenizer.eos_token
model                = AutoModelForCausalLM.from_pretrained(MODEL_ID)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
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

MAX_LENGTH = 256


def preprocess(batch):
    combined = [
        p + " " + c
        for p, c in zip(batch["prompt"], batch["continuation"])
    ]
    encoding = tokenizer(
        combined,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding


print("Tokenizing datasets...")
train_dataset = Dataset.from_pandas(train_df[["prompt", "continuation"]])
val_dataset   = Dataset.from_pandas(val_df[["prompt",   "continuation"]])

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["prompt", "continuation"])
val_dataset   = val_dataset.map(preprocess,   batched=True, remove_columns=["prompt", "continuation"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =============================================================================
# 5. Training Arguments
# =============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    save_total_limit=1,
)

# =============================================================================
# 6. Train
# =============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\nStarting LoRA fine-tuning of GPT-2-Urdu...")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  FP16: {training_args.fp16}")

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapter saved to: {OUTPUT_DIR}")

# =============================================================================
# 7. Post-Training Evaluation on Story Continuation
# =============================================================================

print("\nEvaluating GPT-2-Urdu LoRA on validation set...")
val_sample = val_df.sample(n=min(100, len(val_df)), random_state=42)

model.eval()


def gpt2_continue(prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


predictions = [gpt2_continue(row["prompt"]) for _, row in val_sample.iterrows()]
references  = list(val_sample["continuation"])

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
rougel_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
rougel_avg    = sum(rougel_scores) / len(rougel_scores)

_, _, f1 = bert_score(predictions, references, lang="ur", verbose=False)
bert_avg = f1.mean().item()

print("\n" + "="*45)
print("GPT-2-Urdu LoRA Fine-Tuned - Evaluation Results")
print("="*45)
print(f"  ROUGE-L:   {rougel_avg:.3f}  (thesis target: 0.507)")
print(f"  BERTScore: {bert_avg:.3f}  (thesis target: 0.882)")
print(f"\n  Improvement in ROUGE-L over baseline (0.462): "
      f"{((rougel_avg - 0.462) / 0.462 * 100):+.1f}%")
print(f"  (Thesis reports +9.7% improvement)")

# =============================================================================
# 8. Sample Outputs - LoRA-Tuned Story Continuation
# =============================================================================

STORY_PROMPTS = [
    "ایک دفعہ کا ذکر ہے کہ ایک غریب کسان کا بیٹا شدید بیمار ہو گیا۔",
    "شہر کی پرانی گلیوں میں ایک بوڑھا درخت تھا جس کے نیچے بچے کھیلتے تھے۔",
]

print("\n" + "="*45)
print("Sample LoRA-Tuned Story Continuations")
print("="*45)
for prompt in STORY_PROMPTS:
    print(f"\nPrompt: {prompt}")
    print(f"Output: {gpt2_continue(prompt)}")
