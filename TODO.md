# TODO: Create Small Corpus (1k), Generate Queries, Filter, and Train DPR

## Step 1: Sample 1k Documents from Existing 100k Data
- Use existing processed data in `xuyang/data/msmarco_50/100k/` (or similar for fiqa).
- Sample 1k documents directly from the 100k corpus_filtered_100k_id.tsv or similar file.
- Output: New corpus file with 1k docs (e.g., corpus_1k.jsonl).

## Step 2: Generate Queries for the 1k Corpus
- Use weak supervision or synthetic methods (e.g., based on `zhiyuan/weak_data_loader.py`).
- Generate queries for each document in the 1k corpus.
- Output: queries.jsonl with generated queries.

## Step 3: Filter Queries
- Apply filtering criteria (e.g., remove duplicates, low relevance, diversity checks).
- Use or modify functions in `zhiyuan/data_process.py`.
- Output: Filtered queries.jsonl.

## Step 4: Prepare Data for DPR Training
- Format corpus, queries, and qrels in BEIR/DPR format.
- Ensure qrels/train.tsv, dev.tsv, test.tsv are created or adapted.

## Step 5: Train DPR Model
- Run DPR training script (e.g., `zhiyuan/dpr_eval.py` or similar).
- Use the prepared 1k corpus and filtered queries.

## Step 6: Evaluate Trained Model
- Test the DPR model on dev/test sets.
- Log results.
