"""
Notebook 01: Baseline Evaluation (Objective 1)
-----------------------------------------------
Compares mT5-base and GPT-2-Urdu on:
  - Urdu summarization (ROUGE-L, BERTScore)
  - Urdu story continuation (qualitative outputs)

No prompt engineering or fine-tuning is applied here.
All results correspond to Workflow A in the thesis.

Environment: Google Colab T4 GPU (15 GB VRAM)
"""

# =============================================================================
# 1. Install Dependencies (uncomment when running in Colab)
# =============================================================================

# !pip install -q transformers datasets evaluate rouge-score bert-score sentencepiece

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
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# 3. Load Models and Tokenizers
# =============================================================================

MT5_MODEL_ID  = "google/mt5-base"
GPT2_MODEL_ID = "Imran1/gpt2-urdu-news"

print("\nLoading mT5-base...")
mt5_tokenizer = AutoTokenizer.from_pretrained(MT5_MODEL_ID)
mt5_model     = AutoModelForSeq2SeqLM.from_pretrained(MT5_MODEL_ID).to(DEVICE)
mt5_model.eval()
print(f"  mT5 parameters: {sum(p.numel() for p in mt5_model.parameters()) / 1e6:.0f}M")

print("\nLoading GPT-2-Urdu...")
gpt2_tokenizer            = AutoTokenizer.from_pretrained(GPT2_MODEL_ID)
gpt2_tokenizer.pad_token  = gpt2_tokenizer.eos_token
gpt2_model                = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_ID).to(DEVICE)
gpt2_model.eval()
print(f"  GPT-2-Urdu parameters: {sum(p.numel() for p in gpt2_model.parameters()) / 1e6:.0f}M")

# =============================================================================
# 4. Urdu Summarization - Load XL-Sum Test Set
# =============================================================================

print("\nLoading XL-Sum Urdu test split...")
xlsum = load_dataset("GEM/xlsum", "urdu", split="test", trust_remote_code=True)
xlsum_sample = xlsum.select(range(100))  # evaluate on 100 samples

articles  = [row["document"][:512] for row in xlsum_sample]
summaries = [row["target"] for row in xlsum_sample]

print(f"  Test samples: {len(articles)}")

# =============================================================================
# 5. mT5 Summarization Inference
# =============================================================================

def mt5_summarize(text: str, max_new_tokens: int = 128) -> str:
    prefix = "summarize: "
    inputs = mt5_tokenizer(
        prefix + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = mt5_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)


print("\nRunning mT5 summarization on test set (this may take a few minutes)...")
mt5_preds = [mt5_summarize(a) for a in articles]

# =============================================================================
# 6. GPT-2-Urdu Summarization Inference
# =============================================================================

def gpt2_summarize(text: str, max_new_tokens: int = 128) -> str:
    prompt = text[:300]  # use first portion as prompt for continuation
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
            do_sample=False,
            temperature=1.0,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return gpt2_tokenizer.decode(generated, skip_special_tokens=True)


print("Running GPT-2-Urdu on test set...")
gpt2_preds = [gpt2_summarize(a) for a in articles]

# =============================================================================
# 7. ROUGE-L and BERTScore Evaluation
# =============================================================================

def compute_rougel(predictions: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return round(sum(scores) / len(scores), 3)


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    _, _, f1 = bert_score(predictions, references, lang="ur", verbose=False)
    return round(f1.mean().item(), 3)


print("\nComputing ROUGE-L...")
mt5_rougel  = compute_rougel(mt5_preds,  summaries)
gpt2_rougel = compute_rougel(gpt2_preds, summaries)

print("Computing BERTScore (may take a moment)...")
mt5_bert  = compute_bertscore(mt5_preds,  summaries)
gpt2_bert = compute_bertscore(gpt2_preds, summaries)

results_df = pd.DataFrame({
    "Model":      ["mT5-base", "GPT-2-Urdu"],
    "ROUGE-L":    [mt5_rougel,  gpt2_rougel],
    "BERTScore":  [mt5_bert,    gpt2_bert],
})

print("\n" + "="*45)
print("Table 5.1.1.1 - Summarization Performance (Baseline)")
print("="*45)
print(results_df.to_string(index=False))
print()

# Expected values from thesis:
#   mT5-base:   ROUGE-L=0.505, BERTScore=0.887
#   GPT-2-Urdu: ROUGE-L=0.462, BERTScore=0.873

# =============================================================================
# 8. Urdu Story Continuation - Qualitative Outputs
# =============================================================================

STORY_PROMPTS = [
    "ایک دفعہ کا ذکر ہے کہ ایک غریب کسان کا بیٹا شدید بیمار ہو گیا۔",
    "شہر کی پرانی گلیوں میں ایک بوڑھا درخت تھا جس کے نیچے بچے کھیلتے تھے۔",
    "رات کے اندھیرے میں ایک مسافر نے دور سے روشنی دیکھی اور اس کی طرف چل پڑا۔",
    "ایک چھوٹے سے گاؤں میں ایک استاد تھا جو اپنے طالب علموں کو بہت پیار کرتا تھا۔",
    "پہاڑوں کی چوٹی پر ایک بزرگ رہتا تھا جو بہت حکمت والا تھا۔",
]

print("=" * 45)
print("Story Continuation Outputs - Baseline Models")
print("=" * 45)

for i, prompt in enumerate(STORY_PROMPTS, 1):
    print(f"\n[Prompt {i}]: {prompt}")

    mt5_story = mt5_summarize(prompt, max_new_tokens=150)
    print(f"  mT5-base:   {mt5_story}")

    gpt2_story = gpt2_summarize(prompt, max_new_tokens=150)
    print(f"  GPT-2-Urdu: {gpt2_story}")

# =============================================================================
# 9. Human Evaluation Template
# =============================================================================

print("\n" + "="*45)
print("Human Evaluation Score Reference (from thesis)")
print("="*45)

human_eval = pd.DataFrame({
    "Model":        ["mT5-base", "GPT-2-Urdu"],
    "Fluency":      [8.3, 7.9],
    "Coherence":    [8.1, 7.8],
    "Creativity":   [7.6, 8.2],
    "Cultural Fit": [7.9, 8.5],
})
print(human_eval.to_string(index=False))
print("\nNote: Human scores collected from bilingual review panel (scale 1-10).")
print("GPT-2-Urdu leads in creativity and cultural fit; mT5 leads in fluency and coherence.")
