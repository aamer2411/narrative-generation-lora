"""
Notebook 02: Prompt Engineering (Objective 2 - Part A)
--------------------------------------------------------
Tests four structured prompt strategies on base (untuned) models:
  1. Keyword Prompt
  2. Role + Instruction Prompt
  3. Outline Prompt
  4. Few-Shot Prompt

This corresponds to Workflow B in the thesis.
5 seed prompts x 4 types = 20 prompt configurations per model.

Environment: Google Colab T4 GPU (15 GB VRAM)
"""

# =============================================================================
# 1. Install Dependencies (uncomment when running in Colab)
# =============================================================================

# !pip install -q transformers sentencepiece

# =============================================================================
# 2. Imports
# =============================================================================

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# =============================================================================
# 3. Load Models
# =============================================================================

MT5_MODEL_ID  = "google/mt5-base"
GPT2_MODEL_ID = "Imran1/gpt2-urdu-news"

print("Loading mT5-base...")
mt5_tokenizer = AutoTokenizer.from_pretrained(MT5_MODEL_ID)
mt5_model     = AutoModelForSeq2SeqLM.from_pretrained(MT5_MODEL_ID).to(DEVICE)
mt5_model.eval()

print("Loading GPT-2-Urdu...")
gpt2_tokenizer           = AutoTokenizer.from_pretrained(GPT2_MODEL_ID)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model               = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_ID).to(DEVICE)
gpt2_model.eval()

# =============================================================================
# 4. Seed Prompts (5 story beginnings in Urdu)
# =============================================================================

SEED_PROMPTS = [
    "ایک دفعہ کا ذکر ہے کہ ایک غریب کسان کا بیٹا شدید بیمار ہو گیا۔",
    "شہر کی پرانی گلیوں میں ایک بوڑھا درخت تھا جس کے نیچے بچے کھیلتے تھے۔",
    "رات کے اندھیرے میں ایک مسافر نے دور سے روشنی دیکھی اور اس کی طرف چل پڑا۔",
    "ایک چھوٹے سے گاؤں میں ایک استاد تھا جو اپنے طالب علموں کو بہت پیار کرتا تھا۔",
    "پہاڑوں کی چوٹی پر ایک بزرگ رہتا تھا جو بہت حکمت والا تھا۔",
]

# =============================================================================
# 5. Prompt Engineering Strategies
# =============================================================================

def keyword_prompt(seed: str) -> str:
    """Provide key narrative elements alongside the seed."""
    return (
        f"کہانی کے اہم عناصر: کردار، مشکل، حل\n"
        f"کہانی شروع کریں: {seed}"
    )


def role_instruction_prompt(seed: str) -> str:
    """Assign a narrator role with explicit instruction."""
    return (
        f"آپ ایک تجربہ کار اردو قصہ گو ہیں۔ درج ذیل کہانی کو مکمل کریں اور اسے "
        f"دلچسپ، روانی دار اور ثقافتی لحاظ سے مناسب بنائیں:\n{seed}"
    )


def outline_prompt(seed: str) -> str:
    """Supply a three-act structure for the model to follow."""
    return (
        f"کہانی کا خاکہ:\n"
        f"1. مسئلہ: {seed}\n"
        f"2. کوشش: کردار مشکل سے نکلنے کی کوشش کرتا ہے\n"
        f"3. انجام: ایک امید افزا یا سبق آموز نتیجہ\n\n"
        f"اب اس خاکے کی بنیاد پر مکمل کہانی لکھیں:"
    )


def few_shot_prompt(seed: str) -> str:
    """Provide one example story then ask for a new one."""
    example = (
        "مثال:\n"
        "شروع: ایک بوڑھی عورت جنگل میں راستہ بھول گئی۔\n"
        "کہانی: وہ گھبرائی نہیں بلکہ درختوں کی سمت دیکھ کر راستہ ڈھونڈنے لگی۔ "
        "ایک ہرن نے اسے صحیح سمت دکھائی اور وہ محفوظ گھر پہنچ گئی۔\n\n"
    )
    return example + f"اب درج ذیل شروع سے کہانی مکمل کریں:\nشروع: {seed}\nکہانی:"


PROMPT_STRATEGIES = {
    "Keyword":          keyword_prompt,
    "Role+Instruction": role_instruction_prompt,
    "Outline":          outline_prompt,
    "Few-Shot":         few_shot_prompt,
}

# =============================================================================
# 6. Inference Functions
# =============================================================================

def mt5_generate(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = mt5_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = mt5_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def gpt2_generate(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = gpt2_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = gpt2_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.3,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return gpt2_tokenizer.decode(generated, skip_special_tokens=True)

# =============================================================================
# 7. Run All 20 Prompt Configurations Per Model
# =============================================================================

records = []

print("\n" + "="*55)
print("Workflow B: Prompt Engineering on Base Models")
print("="*55)

for seed_idx, seed in enumerate(SEED_PROMPTS, 1):
    for strategy_name, strategy_fn in PROMPT_STRATEGIES.items():
        engineered = strategy_fn(seed)

        mt5_out  = mt5_generate(engineered)
        gpt2_out = gpt2_generate(engineered)

        records.append({
            "Seed":     seed_idx,
            "Strategy": strategy_name,
            "Model":    "mT5-base",
            "Output":   mt5_out,
        })
        records.append({
            "Seed":     seed_idx,
            "Strategy": strategy_name,
            "Model":    "GPT-2-Urdu",
            "Output":   gpt2_out,
        })

        print(f"\n[Seed {seed_idx} | {strategy_name}]")
        print(f"  Seed prompt : {seed}")
        print(f"  mT5 output  : {mt5_out[:200]}...")
        print(f"  GPT-2 output: {gpt2_out[:200]}...")

df_results = pd.DataFrame(records)
df_results.to_csv("workflow_b_outputs.csv", index=False, encoding="utf-8")
print(f"\nResults saved to workflow_b_outputs.csv ({len(df_results)} rows)")

# =============================================================================
# 8. Summary - Prompt Type Impact
# =============================================================================

print("\n" + "="*55)
print("Prompt Strategy Summary")
print("="*55)
print("""
Based on the thesis findings (Objective 2 - Prompt Engineering):

Strategy         | Key Benefit
-----------------|-------------------------------------------
Keyword          | Anchors characters/plot elements, reduces drift
Role+Instruction | Guides tone and style; more formal register
Outline          | Best narrative structure; three-act arc maintained
Few-Shot         | Highest cultural alignment; model mimics example

Best single-strategy: Outline + Few-Shot (combined) for story continuation.
Prompt engineering alone (Workflow B) improved human scores without any
additional compute cost beyond normal inference.
""")
