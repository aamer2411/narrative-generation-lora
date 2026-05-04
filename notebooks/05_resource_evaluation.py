"""
Notebook 05: Resource Evaluation (Objective 3)
-----------------------------------------------
Measures GPU memory, inference time, and energy consumption across all
four workflows (A-D) for both mT5-base and GPT-2-Urdu.

Workflows:
    A - Baseline: base models + default prompts
    B - Prompt Engineering: base models + structured prompts
    C - LoRA Only: fine-tuned models + default prompts
    D - LoRA + Prompt Engineering: fine-tuned models + structured prompts

Reproduces Table 5.3.1 from the thesis.

Prerequisites:
    - Run 03_lora_finetuning_mt5.py (saves to ./mt5_lora_urdu/)
    - Run 04_lora_finetuning_gpt2.py (saves to ./gpt2_lora_urdu/)

Environment: Google Colab T4 GPU (15 GB VRAM)
"""

# =============================================================================
# 1. Install Dependencies (uncomment when running in Colab)
# =============================================================================

# !pip install -q transformers peft sentencepiece

# =============================================================================
# 2. Imports
# =============================================================================

import time
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE != "cuda":
    print("WARNING: GPU not available. Resource metrics will not reflect real VRAM usage.")

GPU_TDP_WATTS = 70  # T4 GPU typical TDP in watts (used for energy estimate)

# =============================================================================
# 3. Prompt Definitions
# =============================================================================

SEED_PROMPTS = [
    "ایک دفعہ کا ذکر ہے کہ ایک غریب کسان کا بیٹا شدید بیمار ہو گیا۔",
    "شہر کی پرانی گلیوں میں ایک بوڑھا درخت تھا جس کے نیچے بچے کھیلتے تھے۔",
    "رات کے اندھیرے میں ایک مسافر نے دور سے روشنی دیکھی اور اس کی طرف چل پڑا۔",
    "ایک چھوٹے سے گاؤں میں ایک استاد تھا جو اپنے طالب علموں کو بہت پیار کرتا تھا۔",
    "پہاڑوں کی چوٹی پر ایک بزرگ رہتا تھا جو بہت حکمت والا تھا۔",
]

ENGINEERED_PROMPTS = [
    (
        "آپ ایک تجربہ کار اردو قصہ گو ہیں۔ درج ذیل کہانی کو مکمل کریں:\n"
        + p
    )
    for p in SEED_PROMPTS
]

# =============================================================================
# 4. Resource Tracking Utilities
# =============================================================================

def get_gpu_memory_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def reset_gpu_memory():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_peak_gpu_memory_mb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def measure_inference(model_fn, prompts: list[str]) -> dict:
    """Run inference over all prompts and return resource metrics."""
    reset_gpu_memory()

    timings = []
    for prompt in prompts:
        start = time.perf_counter()
        model_fn(prompt)
        timings.append(time.perf_counter() - start)

    avg_time    = sum(timings) / len(timings)
    peak_mem_mb = get_peak_gpu_memory_mb()
    energy      = GPU_TDP_WATTS * avg_time  # W * s = joules, used as W×s estimate

    return {
        "gpu_peak_mb":  round(peak_mem_mb, 0),
        "avg_time_s":   round(avg_time, 2),
        "energy_wxs":   round(energy, 1),
    }

# =============================================================================
# 5. Load All Model Variants
# =============================================================================

MT5_BASE_ID    = "google/mt5-base"
GPT2_BASE_ID   = "Imran1/gpt2-urdu-news"
MT5_LORA_DIR   = "./mt5_lora_urdu"
GPT2_LORA_DIR  = "./gpt2_lora_urdu"

print("\nLoading base mT5-base...")
mt5_tok  = AutoTokenizer.from_pretrained(MT5_BASE_ID)
mt5_base = AutoModelForSeq2SeqLM.from_pretrained(MT5_BASE_ID).to(DEVICE).eval()

print("Loading base GPT-2-Urdu...")
gpt2_tok             = AutoTokenizer.from_pretrained(GPT2_BASE_ID)
gpt2_tok.pad_token   = gpt2_tok.eos_token
gpt2_base            = AutoModelForCausalLM.from_pretrained(GPT2_BASE_ID).to(DEVICE).eval()

print("Loading LoRA-tuned mT5...")
mt5_base_for_lora = AutoModelForSeq2SeqLM.from_pretrained(MT5_BASE_ID)
mt5_lora          = PeftModel.from_pretrained(mt5_base_for_lora, MT5_LORA_DIR).to(DEVICE).eval()

print("Loading LoRA-tuned GPT-2-Urdu...")
gpt2_base_for_lora = AutoModelForCausalLM.from_pretrained(GPT2_BASE_ID)
gpt2_lora          = PeftModel.from_pretrained(gpt2_base_for_lora, GPT2_LORA_DIR).to(DEVICE).eval()

# =============================================================================
# 6. Inference Functions
# =============================================================================

def run_mt5(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def run_gpt2(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# =============================================================================
# 7. Measure All Workflows
# =============================================================================

workflows = []

print("\nMeasuring Workflow A (Baseline - base models + default prompts)...")
m = measure_inference(lambda p: run_mt5(mt5_base,  mt5_tok,  p), SEED_PROMPTS)
workflows.append({"Workflow": "A", "Model": "mT5-base",              **m})
m = measure_inference(lambda p: run_gpt2(gpt2_base, gpt2_tok, p), SEED_PROMPTS)
workflows.append({"Workflow": "A", "Model": "GPT-2-Urdu",            **m})

print("Measuring Workflow B (Prompt Engineering - base models + engineered prompts)...")
m = measure_inference(lambda p: run_mt5(mt5_base,  mt5_tok,  p), ENGINEERED_PROMPTS)
workflows.append({"Workflow": "B", "Model": "mT5 (Prompt Only)",     **m})
m = measure_inference(lambda p: run_gpt2(gpt2_base, gpt2_tok, p), ENGINEERED_PROMPTS)
workflows.append({"Workflow": "B", "Model": "GPT-2 (Prompt Only)",   **m})

print("Measuring Workflow C (LoRA Only - fine-tuned models + default prompts)...")
m = measure_inference(lambda p: run_mt5(mt5_lora,  mt5_tok,  p), SEED_PROMPTS)
workflows.append({"Workflow": "C", "Model": "mT5 (LoRA Only)",       **m})
m = measure_inference(lambda p: run_gpt2(gpt2_lora, gpt2_tok, p), SEED_PROMPTS)
workflows.append({"Workflow": "C", "Model": "GPT-2 (LoRA Only)",     **m})

print("Measuring Workflow D (Prompt + LoRA - fine-tuned models + engineered prompts)...")
m = measure_inference(lambda p: run_mt5(mt5_lora,  mt5_tok,  p), ENGINEERED_PROMPTS)
workflows.append({"Workflow": "D", "Model": "mT5 (Prompt + LoRA)",   **m})
m = measure_inference(lambda p: run_gpt2(gpt2_lora, gpt2_tok, p), ENGINEERED_PROMPTS)
workflows.append({"Workflow": "D", "Model": "GPT-2 (Prompt + LoRA)", **m})

# =============================================================================
# 8. Results Table
# =============================================================================

df = pd.DataFrame(workflows).rename(columns={
    "gpu_peak_mb": "GPU Peak (MB)",
    "avg_time_s":  "Avg. Inference Time (s)",
    "energy_wxs":  "Energy Estimate (W×s)",
})

print("\n" + "="*65)
print("Table 5.3.1 - Resource Usage Metrics Across All Workflows")
print("="*65)
print(df.to_string(index=False))

df.to_csv("workflow_resource_metrics.csv", index=False)
print("\nMetrics saved to workflow_resource_metrics.csv")

# =============================================================================
# 9. Reference Values from Thesis
# =============================================================================

print("\n" + "="*65)
print("Reference: Table 5.3.1 values reported in thesis")
print("="*65)

thesis_reference = pd.DataFrame([
    {"Workflow": "A", "Model": "mT5-base",              "GPU Peak (MB)": 2540, "Avg. Inference Time (s)": 2.1, "Energy (W×s)": 168.8},
    {"Workflow": "A", "Model": "GPT-2-Urdu",            "GPU Peak (MB)": 1115, "Avg. Inference Time (s)": 1.4, "Energy (W×s)":  99.4},
    {"Workflow": "B", "Model": "mT5 (Prompt Only)",     "GPU Peak (MB)": 2610, "Avg. Inference Time (s)": 2.2, "Energy (W×s)": 180.2},
    {"Workflow": "B", "Model": "GPT-2 (Prompt Only)",   "GPU Peak (MB)": 1160, "Avg. Inference Time (s)": 1.5, "Energy (W×s)": 105.0},
    {"Workflow": "C", "Model": "mT5 (LoRA Only)",       "GPU Peak (MB)": 2665, "Avg. Inference Time (s)": 2.3, "Energy (W×s)": 188.4},
    {"Workflow": "C", "Model": "GPT-2 (LoRA Only)",     "GPU Peak (MB)": 1205, "Avg. Inference Time (s)": 1.6, "Energy (W×s)": 112.3},
    {"Workflow": "D", "Model": "mT5 (Prompt + LoRA)",   "GPU Peak (MB)": 2700, "Avg. Inference Time (s)": 2.5, "Energy (W×s)": 197.6},
    {"Workflow": "D", "Model": "GPT-2 (Prompt + LoRA)", "GPU Peak (MB)": 1255, "Avg. Inference Time (s)": 1.7, "Energy (W×s)": 119.8},
])
print(thesis_reference.to_string(index=False))

# =============================================================================
# 10. Recommendation
# =============================================================================

print("\n" + "="*65)
print("Workflow Recommendation")
print("="*65)
print("""
Use Case                   | Recommended Workflow | Model
---------------------------|----------------------|------------------
Best narrative quality     | D (Prompt + LoRA)    | GPT-2-Urdu (8.9)
Institutional/production   | D (Prompt + LoRA)    | mT5-base   (8.8)
Resource-constrained/edge  | B (Prompt Only)      | GPT-2-Urdu
Mobile / low-power         | A (Baseline)         | GPT-2-Urdu

GPT-2-Urdu (Workflow D) uses 53% less GPU memory than mT5 (1255 vs 2700 MB)
while achieving a higher human evaluation score (8.9 vs 8.8).
""")
