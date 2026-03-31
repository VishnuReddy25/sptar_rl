'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''

import torch
from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import argparse
from os.path import join, dirname, abspath
import math
import sys

####
print("Started", flush=True)
print("Started without flush")

zhiyuan_path = dirname(dirname(dirname(dirname(abspath(__file__)))))
if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)

from weak_data_loader import WeakDataLoader

data_dir   = join(zhiyuan_path, "datasets")
raw_dir    = join(data_dir, "raw")
weak_dir   = join(data_dir, "weak")
beir_dir   = join(raw_dir, "beir")
xuyang_dir = join(dirname(zhiyuan_path), "xuyang", "data")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name',    required=False, default="msmarco", type=str)
parser.add_argument('--num_epochs',      required=False, default=2,         type=int)
parser.add_argument('--train_num',       required=False, default=50,        type=int)
parser.add_argument('--weak_num',        required=False, default="5000",    type=str)
parser.add_argument('--product',         required=False, default="cosine",  type=str)
parser.add_argument('--exp_name',        required=False, default="no_aug",  type=str)
# ── New args: direct paths to weak query/qrels files ─────────────────────────
parser.add_argument('--weak_query_file', required=False, default=None,      type=str,
                    help="Direct path to weak queries jsonl file (overrides default path)")
parser.add_argument('--weak_qrels_file', required=False, default=None,      type=str,
                    help="Direct path to weak qrels tsv file (overrides default path)")
args = parser.parse_args()

# ── Model save path ───────────────────────────────────────────────────────────
model_name      = "bert-large-uncased"
model_save_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "output", args.exp_name, str(args.train_num),
    "{}-v1-{}".format(model_name, args.dataset_name)
)
os.makedirs(model_save_path, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
fh = logging.FileHandler(join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[fh, ch]
)
logger = logging.getLogger(__name__)

# ── Data loading ──────────────────────────────────────────────────────────────
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=join(beir_dir, args.dataset_name,
                         f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
        query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
        qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}",
                        f"prompt_tuning_{args.train_num}.tsv")
    ).load_custom()
else:
    # Use directly provided paths if given, otherwise fall back to defaults
    if args.weak_query_file and os.path.exists(args.weak_query_file):
        weak_query_file = args.weak_query_file
        logger.info(f"Using provided weak query file: {weak_query_file}")
    else:
        weak_query_file = join(
            xuyang_dir, "fiqa_50", "500",
            "weak_queries_500_llama_7b_500_fixed_v3_best_llama_prompt_2_filtered_70_filtered_50.jsonl"
        )
        logger.info(f"Using default weak query file: {weak_query_file}")

    if args.weak_qrels_file and os.path.exists(args.weak_qrels_file):
        weak_qrels_file = args.weak_qrels_file
        logger.info(f"Using provided weak qrels file: {weak_qrels_file}")
    else:
        weak_qrels_file = join(
            xuyang_dir, "fiqa_50", "500",
            "weak_train_500_llama_7b_500_fixed_v3_best_llama_prompt_2_filtered_70_filtered_50.tsv"
        )
        logger.info(f"Using default weak qrels file: {weak_qrels_file}")

    corpus, queries, qrels = WeakDataLoader(
        corpus_file=join(beir_dir, args.dataset_name,
                         "corpus_100k_reduced_ratio_20.jsonl"),
        query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
        qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}",
                        f"prompt_tuning_{args.train_num}.tsv"),
        weak_query_file=weak_query_file,
        weak_qrels_file=weak_qrels_file
    ).load_weak_custom()

# ── Dev set ───────────────────────────────────────────────────────────────────
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
    corpus_file=join(beir_dir, args.dataset_name,
                     "corpus_100k_reduced_ratio_20.jsonl"),
    query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
    qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
).load_custom()

# ── Model loading — use existing fine-tuned checkpoint if available ───────────
device = "cuda" if torch.cuda.is_available() else "cpu"

existing_ckpt = os.path.join(
    zhiyuan_path,
    "retriever/dpr/train/output/"
    "llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/"
    "bert-large-uncased-v1-fiqa"
)

if os.path.exists(existing_ckpt):
    logger.info(f"Loading existing fine-tuned DPR checkpoint: {existing_ckpt}")
    model = SentenceTransformer(existing_ckpt, device=device)
    logger.info("Checkpoint loaded — will fine-tune on new GRPO queries")
else:
    logger.warning(
        f"Existing checkpoint not found at {existing_ckpt!r}. "
        f"Falling back to raw {model_name}."
    )
    word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    pooling_model        = models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device=device
    )

print(device)

# ── Retriever & training data ─────────────────────────────────────────────────
retriever        = TrainRetriever(model=model, batch_size=16)
train_samples    = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

# ── Loss ──────────────────────────────────────────────────────────────────────
if args.product == "cosine":
    train_loss      = losses.MultipleNegativesRankingLoss(model=retriever.model)
    score_functions = {'cos_sim': util.cos_sim}
elif args.product == "dot":
    train_loss      = losses.MultipleNegativesRankingLoss(
        model=retriever.model, similarity_fct=util.dot_score
    )
    score_functions = {'dot_score': util.dot_score}

# ── Evaluator ─────────────────────────────────────────────────────────────────
print("IR evaluation without flush")
print("IR evaluation", flush=True)
ir_evaluator = retriever.load_ir_evaluator(
    dev_corpus, dev_queries, dev_qrels, name="dev"
)

# ── Train params ──────────────────────────────────────────────────────────────
num_epochs       = args.num_epochs
evaluation_steps = 0
warmup_steps     = int(
    len(train_samples) * num_epochs / retriever.batch_size * 0.1
)

print(">>> Starting training now...", flush=True)
retriever.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=ir_evaluator,
    epochs=num_epochs,
    output_path=model_save_path,
    warmup_steps=warmup_steps,
    evaluation_steps=evaluation_steps,
    use_amp=True,
    callback=lambda score, epoch, steps: print(
        f"[Epoch {epoch} | Step {steps}] Eval score: {score}", flush=True
    )
)