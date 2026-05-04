# Urdu Narrative Generation via Pre-Trained Language Models with Prompt Engineering and LoRA Fine-Tuning

**MSc Data Science Thesis | Liverpool John Moores University (LJMU)**  
**Author:** Mohammed Aamer | **Supervisor:** Dr. Brett Drury | **August 2025**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![Language](https://img.shields.io/badge/Language-Urdu%20NLP-red)

---

## Overview

This research investigates methods for generating high-quality Urdu narratives (summarization and creative story continuation) under real-world hardware constraints. It compares a **multilingual pre-trained language model** (mT5-base) against an **Urdu-specific language model** (GPT-2-Urdu), enhanced through **prompt engineering** and **LoRA (Low-Rank Adaptation) fine-tuning**.

Urdu represents a meaningful low-resource language challenge: over 70 million native speakers, yet it accounts for only 0.01-0.02% of Common Crawl web data. This work demonstrates that effective, culturally resonant Urdu narrative generation is achievable on a single consumer-grade GPU (Google Colab T4, 15 GB VRAM).

---

## Research Objectives

1. **Baseline Benchmark** - Compare mT5-base and GPT-2-Urdu on identical Urdu summarization and story continuation tasks, with no enhancements.
2. **Enhancement Evaluation** - Measure the impact of structured prompt engineering and LoRA fine-tuning, separately and combined.
3. **Quality-to-Cost Analysis** - Identify the best-performing workflow under constrained hardware (single T4 GPU, 15 GB VRAM).

---

## Approach

### Models
| Model | Type | Parameters |
|---|---|---|
| mT5-base | Multilingual (101 languages) | ~580M |
| GPT-2-Urdu (`Imran1/gpt2-urdu-news`) | Urdu-specific | ~117M |

### Datasets
Five publicly available corpora were combined into a training set of **200,000+ Urdu prompt-continuation pairs**:

| Dataset | Type |
|---|---|
| XL-Sum (Urdu subset) | News summarization (~93K pairs) |
| OpenSubtitles | Conversational / subtitle text |
| JW300 | Parallel Urdu-English sentence pairs |
| CCAligned | Web-scale multilingual text |
| TED Talks | Transcribed speech |

### Workflows Evaluated
| Workflow | Description |
|---|---|
| A - Baseline | Base models, default prompts (no enhancements) |
| B - Prompt Engineering | Base models, structured prompts (Keyword, Role+Instruction, Outline, Few-Shot) |
| C - LoRA Fine-Tuning | LoRA-tuned models, default prompts |
| D - Prompt + LoRA | LoRA-tuned models, structured prompts (full enhancement) |

### LoRA Configuration (mT5)
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
```

### LoRA Configuration (GPT-2-Urdu)
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

---

## Results

### Summarization Performance (Objective 1 vs Objective 2)

| Model | Stage | ROUGE-L | BERTScore |
|---|---|---|---|
| mT5-base | Baseline | 0.505 | 0.887 |
| mT5-base | After LoRA | 0.541 | 0.898 |
| GPT-2-Urdu | Baseline | 0.462 | 0.873 |
| GPT-2-Urdu | After LoRA | 0.507 | 0.882 |

LoRA fine-tuning improved mT5 ROUGE-L by **+7.1%** and GPT-2-Urdu ROUGE-L by **+9.7%**.

### Story Continuation: Human Evaluation Scores (out of 10)

| Model | Fluency | Coherence | Creativity | Cultural Fit |
|---|---|---|---|---|
| mT5-base (Baseline) | 8.3 | 8.1 | 7.6 | 7.9 |
| GPT-2-Urdu (Baseline) | 7.9 | 7.8 | 8.2 | 8.5 |
| mT5 (Workflow D: Prompt + LoRA) | 8.8 (overall) | | | |
| GPT-2-Urdu (Workflow D: Prompt + LoRA) | 8.9 (overall) | | | |

GPT-2-Urdu led in creativity (8.2) and cultural fit (8.5) at baseline. mT5 led in fluency (8.3) and coherence (8.1). Workflow D (combined) achieved the highest overall scores for both models.

### Resource Usage - All Workflows (Objective 3)

| Workflow | Model | GPU Peak (MB) | Avg. Inference Time (s) | Energy (W×s) |
|---|---|---|---|---|
| A - Baseline | mT5-base | 2,540 | 2.1 | 168.8 |
| A - Baseline | GPT-2-Urdu | 1,115 | 1.4 | 99.4 |
| B - Prompt Only | mT5-base | 2,610 | 2.2 | 180.2 |
| B - Prompt Only | GPT-2-Urdu | 1,160 | 1.5 | 105.0 |
| C - LoRA Only | mT5-base | 2,665 | 2.3 | 188.4 |
| C - LoRA Only | GPT-2-Urdu | 1,205 | 1.6 | 112.3 |
| D - Prompt + LoRA | mT5-base | 2,700 | 2.5 | 197.6 |
| D - Prompt + LoRA | GPT-2-Urdu | 1,255 | 1.7 | 119.8 |

GPT-2-Urdu (Workflow D) uses **53% less GPU memory** than mT5 (1,255 MB vs 2,700 MB) while achieving a higher overall human evaluation score (8.9 vs 8.8).

---

## Key Findings

- **Task-dependent strengths:** mT5 outperforms on structured summarization (ROUGE-L 0.505 vs 0.462); GPT-2-Urdu produces more culturally resonant stories (cultural fit 8.5 vs 7.9).
- **Prompt engineering is high-impact, low-cost:** Structured prompts improved narrative quality with negligible additional compute overhead.
- **LoRA is viable on a single GPU:** Full LoRA fine-tuning completed within a 15 GB VRAM Google Colab environment across 200,000+ training pairs.
- **Hybrid workflow (D) is best overall:** Combining LoRA and prompt engineering consistently produced the highest human-rated outputs for both models.
- **GPT-2-Urdu is the efficient choice:** Better quality-to-cost ratio for resource-constrained deployment; fastest inference at 1.4s-1.7s per prompt.
- **mT5 + Workflow D is best for high-quality content:** Achieves strongest automated metrics and is suitable for production summarization pipelines.

---

## Tech Stack

- **Language:** Python 3.10
- **Frameworks:** HuggingFace Transformers, PEFT, Datasets
- **Models:** `google/mt5-base`, `Imran1/gpt2-urdu-news`
- **Fine-tuning:** LoRA via `peft` library
- **Evaluation:** ROUGE-L, BERTScore, Human Evaluation (bilingual panel)
- **Environment:** Google Colab T4 GPU (15 GB VRAM), FP16 mixed precision

---

## Repository Structure

```
urdu-narrative-generation-lora/
├── notebooks/
│   ├── 01_baseline_evaluation.py       # Objective 1: mT5 vs GPT-2-Urdu, no enhancements
│   ├── 02_prompt_engineering.py        # Objective 2a: four prompt strategies, Workflow B
│   ├── 03_lora_finetuning_mt5.py       # Objective 2b: LoRA fine-tuning for mT5-base
│   ├── 04_lora_finetuning_gpt2.py      # Objective 2b: LoRA fine-tuning for GPT-2-Urdu
│   └── 05_resource_evaluation.py       # Objective 3: GPU/memory/energy across all workflows
├── data/
│   ├── prepare_dataset.py              # Downloads and preprocesses all five corpora
│   └── README.md                       # Dataset sources, download instructions
├── requirements.txt
└── README.md
```

### Running Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (downloads ~200K Urdu pairs from five corpora)
python data/prepare_dataset.py

# 3. Run baseline evaluation
python notebooks/01_baseline_evaluation.py

# 4. Run prompt engineering experiments
python notebooks/02_prompt_engineering.py

# 5. Fine-tune models with LoRA
python notebooks/03_lora_finetuning_mt5.py
python notebooks/04_lora_finetuning_gpt2.py

# 6. Evaluate resource usage across all workflows
python notebooks/05_resource_evaluation.py
```

> **Note:** Scripts are designed to run on Google Colab with a T4 GPU. Full fine-tuning (step 5) requires approximately 15 GB VRAM. The full thesis document is not included in this repository in accordance with Liverpool John Moores University academic policy.

---

## Citation

If you reference this work, please cite:

```
Aamer, M. (2025). A Comparative Study on Narrative Generation with Robust Storytelling
Capabilities via Pre-Trained Language Models with Prompt Engineering and LoRA Fine-Tuning.
MSc Data Science Thesis. Liverpool John Moores University (LJMU).
```

---

## Contact

**Mohammed Aamer**  
MSc Data Science, Liverpool John Moores University  
[LinkedIn](https://www.linkedin.com/in/aamer2411/)
