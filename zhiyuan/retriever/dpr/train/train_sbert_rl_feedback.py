'''
Minimal implementation of retriever-aware feedback for SPTAR.
Adds RL-based soft prompt optimization using retrieval performance signals.

Running this script:
python train_sbert_rl_feedback.py --dataset_name fiqa --exp_name rl_feedback_v1
'''

import torch
import torch.nn.functional as F
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

print("Started RL-Feedback Training", flush=True)

zhiyuan_path = dirname(dirname(dirname(dirname(abspath(__file__)))))
if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)

from weak_data_loader import WeakDataLoader

data_dir = join(zhiyuan_path, "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(dirname(zhiyuan_path), "xuyang", "data")


class RLFeedbackLoss(torch.nn.Module):
    """
    Minimal RL-based loss that combines standard ranking loss with retrieval feedback.
    """
    def __init__(self, model, base_loss, feedback_weight=0.1):
        super().__init__()
        self.model = model
        self.base_loss = base_loss
        self.feedback_weight = feedback_weight
        
    def forward(self, sentence_features, labels):
        # Standard ranking loss
        base_loss_value = self.base_loss(sentence_features, labels)
        
        # Compute retrieval quality signal (simplified MRR approximation)
        query_embeddings = sentence_features[0]['sentence_embedding']
        doc_embeddings = sentence_features[1]['sentence_embedding']
        
        # Cosine similarity scores
        scores = torch.mm(query_embeddings, doc_embeddings.transpose(0, 1))
        
        # Compute reciprocal rank reward (higher is better)
        # Positive pairs are on diagonal
        batch_size = scores.shape[0]
        ranks = (scores >= scores.diag().unsqueeze(1)).sum(dim=1).float()
        mrr = (1.0 / ranks).mean()
        
        # RL feedback: maximize MRR (minimize negative MRR)
        rl_loss = -mrr * self.feedback_weight
        
        total_loss = base_loss_value + rl_loss
        
        return total_loss


def train_retriever_with_feedback(args):
    """Train retriever with RL feedback loop"""
    
    # Model save path
    model_name = "bert-base-uncased"
    model_save_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(), 
        "output", 
        args.exp_name, 
        str(args.train_num), 
        f"{model_name}-v1-{args.dataset_name}"
    )
    os.makedirs(model_save_path, exist_ok=True)
    
    # Setup logging
    fh = logging.FileHandler(join(model_save_path, "log.txt"))
    ch = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[fh, ch]
    )
    
    logging.info(f"Starting RL-Feedback Training for {args.dataset_name}")
    logging.info(f"Experiment: {args.exp_name}, Train samples: {args.train_num}")
    
    # Load data - use base_exp_name to load existing weak data
    if args.base_exp_name == "no_aug":
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
            query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
            qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")
        ).load_custom()
    else:
        # Load weak data from base experiment
        weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.base_exp_name}.jsonl")
        weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.base_exp_name}.tsv")
        
        logging.info(f"Loading weak data from: {args.base_exp_name}")
        corpus, queries, qrels = WeakDataLoader(
            corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
            query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
            qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"),
            weak_query_file=weak_query_file,
            weak_qrels_file=weak_qrels_file
        ).load_weak_custom()
    
    # Load dev data
    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
        corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
        query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
        qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
    ).load_custom()
    
    # Initialize model
    word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    
    logging.info(f"Device: {device}")
    
    retriever = TrainRetriever(model=model, batch_size=16)
    
    # Prepare training data
    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    
    # Create RL-enhanced loss
    base_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    train_loss = RLFeedbackLoss(
        model=retriever.model, 
        base_loss=base_loss,
        feedback_weight=args.rl_weight
    )
    
    # Setup evaluation
    logging.info("Setting up IR evaluation", flush=True)
    ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels, name="dev")
    
    # Training configuration
    num_epochs = args.num_epochs
    evaluation_steps = -1  # Evaluate after each epoch
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
    
    logging.info(f"Training config: epochs={num_epochs}, warmup_steps={warmup_steps}")
    logging.info(">>> Starting RL-feedback training...", flush=True)
    
    # Train with feedback
    final_score = retriever.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=num_epochs,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        use_amp=True,
        callback=lambda score, epoch, steps: logging.info(
            f"[Epoch {epoch} | Step {steps}] Eval score: {score}"
        )
    )
    
    logging.info(f"Final evaluation score: {final_score}", flush=True)
    
    # Save experiment metadata
    metadata_path = join(model_save_path, "experiment_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Train samples: {args.train_num}\n")
        f.write(f"Weak samples: {args.weak_num}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"RL weight: {args.rl_weight}\n")
        f.write(f"Final score: {final_score}\n")
    
    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=False, default="fiqa", type=str)
    parser.add_argument('--num_epochs', required=False, default=2, type=int,
                       help="Number of training epochs (use 2 for fast training)")
    parser.add_argument('--train_num', required=False, default=50, type=int)
    parser.add_argument('--weak_num', required=False, default="100k", type=str)
    parser.add_argument('--product', required=False, default="cosine", type=str)
    parser.add_argument('--exp_name', required=False, default="rl_feedback_v1", type=str,
                       help="Experiment name for organizing logs")
    parser.add_argument('--rl_weight', required=False, default=0.1, type=float,
                       help="Weight for RL feedback signal (0.1 for minimal, 0.3 for stronger)")
    args = parser.parse_args()
    
    train_retriever_with_feedback(args)
