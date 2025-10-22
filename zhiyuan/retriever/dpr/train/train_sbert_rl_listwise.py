"""
Train a Bi-Encoder with Listwise loss + optional RL-feedback on BEIR datasets.
Uses MRR-based reward signal to softly guide training.
"""

import os
import sys
import pathlib
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer, losses, evaluation
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever

# ------------------- PATH SETUP -------------------
cwd = os.getcwd()
for p in ["zhiyuan", "xuyang"]:
    full_path = os.path.join(cwd, p)
    if full_path not in sys.path:
        sys.path.append(full_path)

from weak_data_loader import WeakDataLoader

# ------------------- ARGUMENTS -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="scifact", type=str)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--train_num', default=100, type=int)
parser.add_argument('--weak_num', default="5000", type=str)
parser.add_argument('--product', default="cos_sim", type=str)
parser.add_argument('--exp_name', default="rl_feedback_v1", type=str)
parser.add_argument('--model_name', default="bert-base-uncased", type=str)
parser.add_argument('--version', default="v1", type=str)
parser.add_argument('--rl_weight', default=0.1, type=float,
                    help="Weight for RL feedback signal. 0.0 = disable RL.")
args = parser.parse_args()

# ------------------- MODEL SAVE PATH -------------------
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                               "output", args.exp_name, str(args.train_num),
                               args.model_name + '-' + args.version + '-' + args.dataset_name)
os.makedirs(model_save_path, exist_ok=True)

# ------------------- LOGGING -------------------
fh = logging.FileHandler(os.path.join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[fh, ch])

# ------------------- DATA PATHS -------------------
data_dir = os.path.join(cwd, "zhiyuan", "datasets")
raw_dir = os.path.join(data_dir, "raw")
beir_dir = os.path.join(raw_dir, "beir")
xuyang_dir = os.path.join(cwd, "xuyang", "data")

# ------------------- LOAD DATA -------------------
if args.exp_name == "no_aug":
    corpus_file = os.path.join(beir_dir, args.dataset_name,
                               f"corpus_{args.weak_num}_reduced_ratio_20.jsonl")
    query_file = os.path.join(beir_dir, args.dataset_name, "queries.jsonl")
    qrels_file = os.path.join(xuyang_dir,
                              f"{args.dataset_name}_{args.train_num}",
                              f"prompt_tuning_{args.train_num}.tsv")
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_file, query_file=query_file, qrels_file=qrels_file
    ).load_custom()
else:
    weak_query_file = os.path.join(xuyang_dir,
                                   f"{args.dataset_name}_{args.train_num}",
                                   args.weak_num,
                                   f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = os.path.join(xuyang_dir,
                                   f"{args.dataset_name}_{args.train_num}",
                                   args.weak_num,
                                   f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus_file = os.path.join(beir_dir, args.dataset_name,
                               f"corpus_{args.weak_num}_reduced_ratio_20.jsonl")
    query_file = os.path.join(beir_dir, args.dataset_name, "queries.jsonl")
    qrels_file = os.path.join(xuyang_dir,
                              f"{args.dataset_name}_{args.train_num}",
                              f"prompt_tuning_{args.train_num}.tsv")
    corpus, queries, qrels = WeakDataLoader(
        corpus_file=corpus_file,
        query_file=query_file,
        qrels_file=qrels_file,
        weak_query_file=weak_query_file,
        weak_qrels_file=weak_qrels_file
    ).load_weak_custom()

# Dev set
dev_corpus_file = os.path.join(beir_dir, args.dataset_name,
                               f"corpus_{args.weak_num}_reduced_ratio_20.jsonl")
dev_query_file = os.path.join(beir_dir, args.dataset_name, "queries.jsonl")
dev_qrels_file = os.path.join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
    corpus_file=dev_corpus_file,
    query_file=dev_query_file,
    qrels_file=dev_qrels_file
).load_custom()

# ------------------- MODEL -------------------
word_embedding_model = models.Transformer(args.model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# ------------------- BASE LOSS (LISTWISE) -------------------
base_loss = losses.MultipleNegativesRankingLoss(model=model)

# ------------------- RL FEEDBACK WRAPPER -------------------
class RLFeedbackLoss(nn.Module):
    def __init__(self, model, base_loss, feedback_weight=0.1):
        super().__init__()
        self.model = model
        self.base_loss = base_loss
        self.feedback_weight = feedback_weight

    def forward(self, sentence_features, labels):
        base_loss_val = self.base_loss(sentence_features, labels)

        # Compute retrieval-like reward (MRR approximation)
        q_emb = sentence_features[0]['sentence_embedding']
        d_emb = sentence_features[1]['sentence_embedding']
        scores = torch.mm(q_emb, d_emb.transpose(0, 1))

        batch_size = scores.shape[0]
        ranks = (scores >= scores.diag().unsqueeze(1)).sum(dim=1).float()
        mrr = (1.0 / ranks).mean()

        rl_loss = -mrr * self.feedback_weight
        total_loss = base_loss_val + rl_loss
        return total_loss

# If rl_weight is zero, just use base loss
if args.rl_weight > 0:
    train_loss = RLFeedbackLoss(model, base_loss, feedback_weight=args.rl_weight)
else:
    train_loss = base_loss

# ------------------- EVALUATOR -------------------
dev_evaluator = evaluation.InformationRetrievalEvaluator(
    dev_queries, dev_corpus, dev_qrels,
    name=args.dataset_name, main_score_function=args.product
)

# ------------------- TRAINING SAMPLES -------------------
retriever = TrainRetriever(model=model, batch_size=32)
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

# ------------------- TRAINING -------------------
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    logging.info(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

logging.info("Starting RL+Listwise training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=args.num_epochs,
    output_path=model_save_path,
    warmup_steps=100,
    use_amp=False,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=len(train_dataloader),
    evaluation_steps=1000,
    save_best_model=True
)

logging.info(f"Training complete. Model saved to {model_save_path}")
