# SPTAR — Phase 2 GRPO Pipeline

**Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models**  
This version covers the **Phase 2 GRPO soft-prompt improvement loop** only.  
ColBERT, BM25, and BM25CE are not included.

---

## What is SPTAR?

SPTAR is a pipeline that uses soft prompt tuning on an LLM (LLaMA-2-7B) to automatically generate weak document–query pairs, which are then used to train a dense retriever (DPR). The pipeline has two phases:

- **Phase 1** — Initial soft prompt training to generate a seed set of weak queries
- **Phase 2 (this repo)** — GRPO reinforcement loop that improves the soft prompt using DPR-based reward signals, producing a larger and higher-quality query pool

---

## Repository Structure

```
.
├── xuyang/                         # Soft prompt tuning modules
│   ├── llm_models/                 # Phase 1 .npy soft-prompt weights
│   └── default_prompt.py           # Fixed few-shot prompt templates
├── zhiyuan/                        # DPR retriever modules
│   ├── datasets/raw/beir/          # BEIR datasets (fiqa, msmarco, etc.)
│   └── retriever/dpr/train/        # DPR training scripts
├── grpo_phase2_v2.py               # ← Main Phase 2 training script
├── output/grpo_phase2/             # Generated outputs (queries, adapters)
├── logs/                           # Auto-generated training logs
└── environment.yml                 # Conda environment spec
```

---

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate sptar
```

If you see this error:
```
nvidia/cublas/lib/libcublas.so.11: symbol cublasLtGetStatusString ...
```
Run:
```bash
pip uninstall nvidia_cublas_cu11
```

If you see a `setuptools` / `packaging` import error:
```bash
pip install setuptools==69.5.1
```

### 2. Install modified packages

```bash
cd package/beir
pip install -e .

cd package/sentence-transformers
pip install -e .
```

---

## Data Preparation

Download BEIR datasets and generate the required corpus files:

```bash
python zhiyuan/download.py
python zhiyuan/data_process.py
```

This generates three files per dataset under `zhiyuan/datasets/raw/beir/<dataset>/`:

| File | Purpose |
|---|---|
| `corpus_filtered.jsonl` | All unlabelled documents |
| `corpus_5000_reduced_ratio_20.jsonl` | Small eval corpus (used when weak pairs ≤ 5k) |
| `corpus_100k_reduced_ratio_20.jsonl` | Larger eval corpus (used when weak pairs ≥ 100k) |

Alternatively, download our pre-processed datasets from Google Drive and place them under `zhiyuan/`.

---

## Phase 2 GRPO Training

### What it does

1. Loads the LLaMA-2-7B model with the Phase 1 soft-prompt weights
2. Each epoch: selects 300 informative docs using DPR embedding variance ranking
3. Generates a group of candidate queries per doc using the soft prompt
4. Scores queries with a composite reward (relevance + retrievability + specificity)
5. Updates the soft prompt via GRPO (clipped policy gradient + KL penalty)
6. Periodically retrains the DPR retriever on the accumulated query pool
7. Saves the best adapter based on NDCG@10 on a held-out eval set

### Run command

```bash
python grpo_phase2_v2.py \
    --peft_model_id   <path_to_phase1_peft_adapter> \
    --dpr_ckpt        <path_to_dpr_checkpoint> \
    --weak_queries    <path_to_phase1_weak_queries.jsonl> \
    --weak_qrels      <path_to_phase1_weak_train.tsv> \
    --corpus          zhiyuan/datasets/raw/beir/fiqa/corpus_filtered.jsonl \
    --eval_corpus     zhiyuan/datasets/raw/beir/fiqa/corpus_5000_reduced_ratio_20.jsonl \
    --dev_queries     zhiyuan/datasets/raw/beir/fiqa/dev_queries.jsonl \
    --dev_qrels       zhiyuan/datasets/raw/beir/fiqa/dev_qrels.tsv \
    --dataset_name    fiqa_50 \
    --phase1_npy      ./xuyang/llm_models/v1_fiqa_50_llama-7b_llama-7b_CAUSAL_LM_TEXT_50_50_3_2023-06-04_0 \
    --output_dir      output/grpo_phase2 \
    --grpo_group_size 6 \
    --max_new_tokens  40 \
    --lr              5e-3 \
    --kl_coeff        0.02 \
    --clip_ratio      0.2 \
    --temperature     0.5 \
    --grpo_epochs     2 \
    --docs_per_epoch  300 \
    --load_in_4bit
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--peft_model_id` | required | Path to Phase 1 PEFT adapter directory |
| `--dpr_ckpt` | required | Path to pre-trained DPR/SentenceTransformer checkpoint |
| `--phase1_npy` | see above | Path to Phase 1 `.npy` soft-prompt weights |
| `--dataset_name` | `fiqa_50` | One of `fiqa_50`, `ms_50`, `hotpotqa_50`, `fever_50` |
| `--docs_per_epoch` | `300` | Docs selected per epoch via DPR ranking |
| `--grpo_group_size` | `6` | Candidate queries generated per document |
| `--grpo_epochs` | `2` | Number of training epochs |
| `--kl_coeff` | `0.02` | KL penalty weight (auto-adjusted during training) |
| `--dpr_retrain_every` | `200` | Retrain DPR every N processed docs |
| `--load_in_4bit` | True | Load LLM in 4-bit NF4 (saves ~14GB VRAM) |

---

## Outputs

After training, `output/grpo_phase2/` contains:

| File / Folder | Description |
|---|---|
| `best_peft_adapter/` | Adapter checkpoint with best NDCG@10 |
| `final_peft_adapter/` | Adapter at end of training |
| `peft_epoch{N}/` | Adapter saved after each epoch |
| `final_weak_queries.jsonl` | Full accumulated query pool |
| `final_weak_train.tsv` | Full accumulated qrels |
| `queries_step{N}.jsonl` | Query pool snapshot at step N |
| `summary.json` | Best NDCG@10, history, pool size |
| `logs/grpo_phase2_*.log` | Full training log |

---

## Final DPR Training

After Phase 2 completes, train the final DPR retriever on the full query pool:

```bash
python zhiyuan/dpr_eval.py \
    --dataset_name fiqa \
    --version v1 \
    --gpu_id 0 \
    --train_num 50 \
    --weak_num <total_queries_in_pool> \
    --exp_names grpo_phase2
```

The exact command is also printed at the end of every training run.

---

## Query & Document Filtering (Phase 2 v2)

The pipeline applies strict filters to ensure only clean, meaningful data enters training:

**Document filters** — skip docs with fewer than 50 words or less than 50% alphabetic tokens

**Query filters (post-generation)** — reject queries that:
- Contain any digit
- Are shorter than 4 words or longer than 20 words
- Have more than 10% non-alphabetic characters

**Reward filters** — skip the entire document if all generated queries score zero reward

**Reward components:**

| Component | Weight | Description |
|---|---|---|
| Relevance | 0.4 | Token overlap between query and document |
| Retrievability | 0.4 | Reciprocal rank of the source doc when retrieved by the query |
| Specificity | 0.2 | Query length, structure, and vocabulary diversity |

---

## Citing

If you find SPTAR helpful, please cite:

```bibtex
@article{DBLP:journals/kbs/PengWWF25,
  author  = {Zhiyuan Peng and Xuyang Wu and Qifan Wang and Yi Fang},
  title   = {Soft prompt tuning for augmenting dense retrieval with large language models},
  journal = {Knowl. Based Syst.},
  volume  = {309},
  pages   = {112758},
  year    = {2025},
  doi     = {10.1016/J.KNOSYS.2024.112758}
}
```
