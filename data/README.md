# Dataset Information

This folder holds the preprocessed training and evaluation data used in this research. The raw corpora are publicly available and must be downloaded and preprocessed using `prepare_dataset.py` before running the fine-tuning notebooks.

## Source Corpora

| Dataset | Source | Urdu Content | HuggingFace ID |
|---|---|---|---|
| XL-Sum | BBC News (44 languages) | ~93K article-summary pairs | `GEM/xlsum` |
| OpenSubtitles | Movie/TV subtitles | ~3.5M Urdu-English sentence pairs | `open_subtitles` |
| JW300 | Jehovah's Witness publications | ~100K Urdu-English pairs | `opus100` |
| CCAligned | Common Crawl web text | ~1M+ Urdu-English document pairs | `cc_aligned` |
| TED Talks | TED talk transcripts | ~200K Urdu-English sentence pairs | `ted_talks_iwslt` |

The combined dataset used for fine-tuning contains **200,000+ Urdu prompt-continuation pairs** sampled and balanced across all five corpora.

## Generating the Dataset

```bash
python data/prepare_dataset.py
```

This produces:
- `data/urdu_train.csv` - Training split (90%)
- `data/urdu_val.csv` - Validation split (10%)

Each CSV has two columns: `prompt` and `continuation`.

## Notes

- For the **summarization task**, XL-Sum Urdu articles are used directly: the article body is the prompt and the summary is the continuation.
- For the **story continuation task**, Urdu text from the other corpora is split at a natural boundary (roughly the first 30-40% of tokens) into prompt and continuation.
- All text is kept in its original Urdu script (right-to-left, Unicode).
- No translation is applied; all data is native Urdu.
