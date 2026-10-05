# Amazon ML Challenge 2026 — Business Entity Resolution

## Author

**Yash Namdeo** — AI/ML Engineer
- GitHub: [@yash112005](https://github.com/yash112005)
- LinkedIn: [www.linkedin.com/in/yash-namdeo-48412531a]
  
## Overview

In large-scale commercial platforms, business identity data arrives from multiple 
independent sources — each contributing partial, noisy fragments about the same 
real-world entities. This project solves the **Entity Resolution** problem: given 
business records from 3 independent sources, determine which records across 
sources refer to the same real-world business entity.

**Final F0.5 Score (local validation): 0.8875**

## Problem Statement

- **Source 1**: Deduplicated reference source — find matches for every entity here
- **Source 2 & 3**: Candidate sources — matching records need to be found in these
- A Source 1 entity may have **zero, one, or many** matches in Source 2/3
- Noisy data patterns: name abbreviations (Corp/Corporation, Pvt/Private), legal 
  suffix inconsistencies, typos, word-order transpositions, incomplete addresses, 
  landmark-based address references
- **Scale**: Source 1 ~2.2M records, Source 2 ~5M records, Source 3 ~5.2M records — 
  roughly 12.5 million business records in total

## Evaluation Metric

**F0.5 Score** (precision-weighted F-beta score) — penalizes false merges (incorrectly 
matching two different businesses) **2x harder** than missed matches (false negatives). 
This shaped the entire pipeline's priority: only commit to confident matches, and 
leave ambiguous cases unmatched rather than risk a false merge.



## Approach — Two-Stage Pipeline

At this scale (12.5M+ records), brute-force comparison (checking every record 
against every other record) is computationally infeasible. The problem was 
therefore split into two stages:

### Stage 1 — Blocking / Candidate Generation
**File: `dataset_analysis_partA.ipynb`** (Candidate Generation)

- Text cleaning: lowercased names/addresses, stripped symbols, normalized whitespace
- Stopword filtering: excluded generic terms ("Ltd", "Inc", "and", "Private") from 
  the blocking key — without this, these words were collapsing records into single 
  giant blocks (97,000+ records in some cases)
- Blocking key: country + the most significant word in the business name 
  (alphabetically sorted, stopwords and short words excluded)
- Normal-sized blocks: resolved via a simple dictionary/groupby lookup
- Oversized blocks (>3,000 records): re-ranked using TF-IDF (character n-grams, 
  length 2-4) + Nearest Neighbors (cosine similarity), scoped only to that block 
  — avoiding a full-dataset TF-IDF pass
- Output: `candidate_pairs.tsv` — top candidate matches for every Source 1 entity

### Stage 2 — Matching Model
**File: `model_train.ipynb`**

- Engineered similarity features on candidate pairs (via RapidFuzz):
  - Name similarity: ratio, token-sort-ratio, token-set-ratio, partial-ratio
  - Address similarity: same set of metrics
  - Length differences, country match
- Classifier: **LightGBM** (gradient boosting) — efficient at this data scale, 
  with `is_unbalance=True` to handle class imbalance
- Threshold tuning: decision threshold tuned high (precision-favoring) to directly 
  optimize for F0.5, since a false merge is costlier than a missed match

### End-to-End Pipeline
**File: `main.ipynb`** — runs the complete flow from raw data to final predictions

## Engineering Challenges Solved

At this data scale, the harder problems were in data engineering, not modeling:

- **Memory management**: Loading 5M+ row files triggered MemoryErrors — resolved 
  via chunked reading (`chunksize`) and dtype optimization (`category` dtype)
- **Checkpoint/resume system**: Oversized-block processing could take hours; losing 
  progress to a session crash (common on Colab) was costly. Implemented batch-wise 
  checkpointing — progress saved to disk after every batch, resumable from the 
  last completed batch on crash
- **I/O optimization**: Re-reading the full 12.5M+ row files for every single block 
  was too slow — batching 10 blocks per file-read pass cut file-reading by 10x
- **Data integrity debugging**: Incomplete Google Drive uploads were silently 
  truncating files (a 2.2M-row file was loading as 132K rows) — root-caused and 
  fixed through systematic verification at each pipeline stage

## Results

| Metric | Value |
|---|---|
| Blocking recall (sample) | ~85-90% |
| Average candidates per entity | 32.7 |
| Final F0.5 Score (local validation) | 0.8875 |

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.x | Core language |
| **Pandas** | 3.0.6 | Data manipulation (CSV/TSV handling, dataframes) |
| **NumPy** | 2.5.3 | Numerical operations |
| **Scikit-learn** | 1.9.1 | TF-IDF Vectorizer, Nearest Neighbors (blocking stage) |
| **LightGBM** | 4.7.0 | Gradient boosting classifier (matching stage) |
| **RapidFuzz** | 3.14.6 | Fuzzy string similarity (name/address matching) |
| **SciPy** | 1.18.1 | Scikit-learn dependency (sparse matrix operations) |
| **Matplotlib** | 3.11.2 | Data visualization (EDA) |
| **Seaborn** | 0.13.2 | Statistical visualization |
| **Jupyter / ipykernel** | — | Notebook execution environment (`.ipynb` files) |

**Development environment:** Google Colab / VS Code  

## Dataset

This project uses the dataset provided exclusively to registered participants 
of Amazon ML Challenge 2026 (Business Entity Resolution Challenge) via the 
official competition platform. 

**The dataset is not publicly available and is not included in this repository**, 
in compliance with the competition's data usage terms. This repository contains 
only the processing pipeline and modeling code.

To run this code, you would need access to a similarly structured dataset with 
the following schema:
- `entity_id`, `business_name`, `business_address`, `country` (source files)
- `source1_entity_id`, `matched_entity_ids` (ground truth)

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn lightgbm rapidfuzz`
2. Run `python candidate_generation.py` — generates `candidate_pairs.tsv`
3. Run `model_train.ipynb` — trains the model and produces final predictions
4. Run `main.ipynb` to see the complete pipeline end-to-end

## Key Learnings

In real-world, large-scale ML problems, model training is only one part of the 
work — a significant share (arguably 60-70%) is data engineering: memory-safe 
processing, crash-resilient pipelines, and efficient I/O. This project was built 
to demonstrate exactly that balance.

## Future Improvements

- Multiple blocking keys (first-word + last-word combination) to improve recall further
- Ensemble of LightGBM + XGBoost for more robust predictions
- Hyperparameter tuning via Optuna for the matching model

## Note

The reported score is based on local validation. Official competition submission 
wasn't completed before the deadline, but the full pipeline was built end-to-end 
for learning and portfolio purposes.








