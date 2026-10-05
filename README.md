# Amazon ML Challenge 2026 — Business Entity Resolution

## Overview

Large-scale commercial platforms mein business identity data multiple independent 
sources se aata hai — har source thoda noisy aur inconsistent format mein data 
deta hai. Ye project solve karta hai **Entity Resolution** problem: business 
records ke 3 alag sources se, pata lagana ki kaunse records asal mein ek hi 
real-world business entity ko represent karte hain.

**Final F0.5 Score (local validation): 0.8875**

## Problem Statement

- **Source 1**: Deduplicated reference source — har entity ke liye match dhoondhna hai
- **Source 2 & 3**: Candidate sources — inme se matching records dhoondhne hain
- Ek Source 1 entity ka Source 2/3 mein **zero, one, ya multiple** matches ho sakte hain
- Data mein noise: name abbreviations (Corp/Corporation, Pvt/Private), legal suffix 
  inconsistencies, typos, word-order variations, incomplete addresses, landmark-based 
  address references
- **Scale**: Source 1 ~22 lakh records, Source 2 ~50 lakh records, Source 3 ~52 lakh 
  records — total ~1.25 crore business records

## Evaluation Metric

**F0.5 Score** (precision-weighted F-beta score) — false merges (do different 
businesses ko ek bata dena) ko false negatives (match miss karna) se **2x zyada 
penalize** karta hai. Isliye poore pipeline mein priority rahi: confident matches 
hi final list mein rakho, ambiguous cases ko khali chhodo.


## Approach — Two-Stage Pipeline

Itne bade scale (1.25 crore+ records) pe brute-force comparison (har record ko 
har doosre record se compare karna) practically impossible hai. Isliye kaam ko 
2 stages mein divide kiya:

### Stage 1 — Blocking / Candidate Generation
**File: `dataset_analysis_partA.ipynb`** (Candidate Generation)

- Text cleaning: naam/address lowercase, symbols hatana, extra spaces normalize karna
- Generic/stopword filtering: "Ltd", "Inc", "and", "Private" jaise common words ko 
  blocking key se exclude kiya — warna ye words single giant blocks (97,000+ records 
  tak) bana rahe the
- Blocking key: country + naam ka sabse significant word (alphabetically sorted, 
  stopwords aur chhote words exclude karke)
- Normal-size blocks: simple dictionary/groupby lookup se candidates nikale
- Oversized blocks (>3000 records): TF-IDF (character n-grams, 2-4 length) + 
  Nearest Neighbors (cosine similarity) se sirf us block ke andar re-ranking, 
  taaki poore dataset pe TF-IDF na chalana pade
- Output: `candidate_pairs.tsv` — har Source 1 entity ke liye uske top candidates

### Stage 2 — Matching Model
**File: `model_train1.ipynb`**

- Candidate pairs pe similarity features banaye (RapidFuzz library se):
  - Name similarity: ratio, token-sort-ratio, token-set-ratio, partial-ratio
  - Address similarity: same metrics
  - Length differences, country match
- Classifier: **LightGBM** (gradient boosting) — bade data pe efficient, class 
  imbalance handle karne ke liye `is_unbalance=True`
- Threshold tuning: F0.5 metric ko directly optimize karne ke liye decision 
  threshold ko high rakha (precision-priority), kyunki false merge zyada costly hai

### End-to-End Pipeline
**File: `main.ipynb`** — poora flow raw data se final predictions tak

## Engineering Challenges Solved

Is scale ke data ke saath kaam karte waqt, pure ML se zyada **data engineering** 
challenges aaye:

- **Memory management**: 50+ lakh row files load karte waqt MemoryError — fix kiya 
  chunked reading (`chunksize` parameter) aur dtype optimization (`category` dtype) se
- **Checkpoint/Resume system**: Oversized block processing ghanton le sakta tha; 
  agar session crash ho (jo Colab mein common hai), progress khona bahut costly 
  hota. Isliye batch-wise checkpointing implement kiya — har batch ke baad progress 
  disk pe save, crash hone pe wahi se resume
- **I/O optimization**: Har block ke liye poori 1 crore+ row file dobara padhna 
  bahut slow tha — batch processing (10 blocks ek saath) se file-reading 10x kam 
  ki gayi
- **Data integrity debugging**: Incomplete Google Drive uploads ki wajah se files 
  silently truncate ho rahi thi (22 lakh rows ki jagah 1.3 lakh load ho rahi thi) — 
  systematic debugging se root-cause pakda aur fix kiya

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

1. Dependencies install karo: `pip install pandas numpy scikit-learn lightgbm rapidfuzz`
2. `python candidate_generation.py` chalao — `candidate_pairs.tsv` generate hoga
3. `model_train.ipynb` chalao — model train hoga aur final predictions banenge
4. Pura flow ek saath dekhne ke liye `main.ipynb` chalao

## Key Learnings

Real-world, large-scale ML problems mein model training sirf ek hissa hai — 
poora kaam ka bada hissa (shayad 60-70%) **data engineering** hota hai: memory-safe 
processing, crash-resilient pipelines, aur efficient I/O. Ye project isi balance 
ko practically demonstrate karta hai.

## Note

The score reported here is based on local validation. Official competition 
submission was not completed before the deadline, but the full pipeline was 
built end-to-end for learning and skill development purposes.
