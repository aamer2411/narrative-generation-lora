"""
Dataset Preparation
-------------------
Downloads and preprocesses five publicly available Urdu corpora into a unified
prompt-continuation format for fine-tuning mT5 and GPT-2-Urdu.

Output:
    data/urdu_train.csv  -- 90% split
    data/urdu_val.csv    -- 10% split

Run:
    python data/prepare_dataset.py
"""

import os
import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(OUTPUT_DIR, "urdu_train.csv")
VAL_PATH   = os.path.join(OUTPUT_DIR, "urdu_val.csv")

TARGET_PAIRS = 200_000
SEED         = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_prompt_continuation(text: str, split_ratio: float = 0.35) -> tuple[str, str]:
    """Split Urdu text into a prompt (first ~35%) and continuation (rest)."""
    words = text.split()
    if len(words) < 10:
        return None, None
    split_at = max(5, int(len(words) * split_ratio))
    return " ".join(words[:split_at]), " ".join(words[split_at:])


def is_valid_urdu(text: str, min_chars: int = 30) -> bool:
    """Rough check: text contains Urdu Unicode characters."""
    urdu_range = re.compile(r"[؀-ۿ]")
    return len(text) >= min_chars and bool(urdu_range.search(text))


def collect_pairs(records: list[tuple[str, str]]) -> list[dict]:
    """Filter and return valid prompt-continuation dicts."""
    out = []
    for prompt, cont in records:
        if prompt and cont and is_valid_urdu(prompt) and is_valid_urdu(cont):
            out.append({"prompt": prompt.strip(), "continuation": cont.strip()})
    return out


# ---------------------------------------------------------------------------
# Per-corpus loaders
# ---------------------------------------------------------------------------

def load_xlsum(n: int = 60_000) -> list[dict]:
    """XL-Sum Urdu: article = prompt, summary = continuation."""
    print("Loading XL-Sum (Urdu)...")
    ds = load_dataset("GEM/xlsum", "urdu", split="train", trust_remote_code=True)
    pairs = []
    for row in tqdm(ds, total=min(n, len(ds))):
        doc  = row.get("document", "")
        summ = row.get("target", "")
        if is_valid_urdu(doc) and is_valid_urdu(summ):
            # Truncate long articles to first 400 chars for a usable prompt
            pairs.append({"prompt": doc[:400].strip(), "continuation": summ.strip()})
        if len(pairs) >= n:
            break
    print(f"  XL-Sum pairs collected: {len(pairs)}")
    return pairs


def load_opensubtitles(n: int = 50_000) -> list[dict]:
    """OpenSubtitles: Urdu side of en-ur parallel corpus, split into prompt/cont."""
    print("Loading OpenSubtitles (ur)...")
    try:
        ds = load_dataset(
            "open_subtitles", lang1="en", lang2="ur", split="train",
            streaming=True, trust_remote_code=True
        )
        raw = []
        for row in tqdm(ds, total=n):
            text = row.get("translation", {}).get("ur", "")
            if text:
                raw.append(text)
            if len(raw) >= n * 2:
                break
    except Exception as e:
        print(f"  OpenSubtitles load error: {e}. Skipping.")
        return []

    records = [split_prompt_continuation(t) for t in raw]
    pairs   = collect_pairs(records)[:n]
    print(f"  OpenSubtitles pairs collected: {len(pairs)}")
    return pairs


def load_jw300(n: int = 30_000) -> list[dict]:
    """JW300 via opus100: Urdu-English, take Urdu side and split."""
    print("Loading JW300 via opus100 (ur-en)...")
    try:
        ds = load_dataset("opus100", "en-ur", split="train", streaming=True, trust_remote_code=True)
        raw = []
        for row in tqdm(ds, total=n):
            text = row.get("translation", {}).get("ur", "")
            if text:
                raw.append(text)
            if len(raw) >= n * 2:
                break
    except Exception as e:
        print(f"  JW300/opus100 load error: {e}. Skipping.")
        return []

    records = [split_prompt_continuation(t) for t in raw]
    pairs   = collect_pairs(records)[:n]
    print(f"  JW300 pairs collected: {len(pairs)}")
    return pairs


def load_ccaligned(n: int = 40_000) -> list[dict]:
    """CCAligned: Urdu-English web text, take Urdu side and split."""
    print("Loading CCAligned (ur-en)...")
    try:
        ds = load_dataset("cc_aligned", "ur-en", split="train", streaming=True, trust_remote_code=True)
        raw = []
        for row in tqdm(ds, total=n):
            text = row.get("translation", {}).get("ur", "")
            if text:
                raw.append(text)
            if len(raw) >= n * 2:
                break
    except Exception as e:
        print(f"  CCAligned load error: {e}. Skipping.")
        return []

    records = [split_prompt_continuation(t) for t in raw]
    pairs   = collect_pairs(records)[:n]
    print(f"  CCAligned pairs collected: {len(pairs)}")
    return pairs


def load_ted_talks(n: int = 20_000) -> list[dict]:
    """TED Talks IWSLT: Urdu side of en-ur, split into prompt/cont."""
    print("Loading TED Talks (ur)...")
    try:
        ds = load_dataset(
            "ted_talks_iwslt", language_pair=("en", "ur"),
            year="2014", split="train", trust_remote_code=True
        )
        raw = [row.get("translation", {}).get("ur", "") for row in ds]
    except Exception as e:
        print(f"  TED Talks load error: {e}. Skipping.")
        return []

    records = [split_prompt_continuation(t) for t in raw]
    pairs   = collect_pairs(records)[:n]
    print(f"  TED Talks pairs collected: {len(pairs)}")
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_pairs = []
    all_pairs.extend(load_xlsum())
    all_pairs.extend(load_opensubtitles())
    all_pairs.extend(load_jw300())
    all_pairs.extend(load_ccaligned())
    all_pairs.extend(load_ted_talks())

    print(f"\nTotal raw pairs: {len(all_pairs)}")

    df = pd.DataFrame(all_pairs).drop_duplicates(subset=["prompt"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if len(df) > TARGET_PAIRS:
        df = df.iloc[:TARGET_PAIRS]

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_PATH,   index=False, encoding="utf-8")

    print(f"\nDataset saved:")
    print(f"  Train: {len(train_df):,} pairs -> {TRAIN_PATH}")
    print(f"  Val:   {len(val_df):,} pairs  -> {VAL_PATH}")


if __name__ == "__main__":
    main()
