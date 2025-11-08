"""
Advanced Retrieval Evaluator for RL Rewards

Computes comprehensive retrieval metrics and converts them to RL reward signals.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from zhiyuan.data_process import load_dl, merge_queries, extract_results


class RetrievalEvaluator:
    """Advanced retrieval evaluator for RL feedback"""
    
    def __init__(self, model_path: str, dataset_name: str = "fiqa", corpus_size: int = 500):
        self.dataset_name = dataset_name
        self.corpus_size = corpus_size
        self.model_path = model_path
        
        # Load model
        self.model = SentenceTransformer(model_path)
        self.evaluator = EvaluateRetrieval(self.model, k_values=[1, 3, 5, 10, 100])
        
        # Load data
        self._load_data()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_data(self):
        """Load dataset based on configuration"""
        data_dir = Path(__file__).parent.parent.parent / "zhiyuan" / "datasets"
        beir_dir = data_dir / "raw" / "beir"
        
        if self.dataset_name == "msmarco":
            corpus, queries, qrels = GenericDataLoader(str(beir_dir / self.dataset_name)).load(split="dev")
            queries_19, qrels_19, qrels_binary_19 = load_dl(str(beir_dir / "TREC_DL_2019"))
            queries_20, qrels_20, qrels_binary_20 = load_dl(str(beir_dir / "TREC_DL_2020"))
            self.queries = merge_queries(queries, queries_19, queries_20)
            self.qrels = qrels_19
        else:
            corpus, self.queries, self.qrels = GenericDataLoader(str(beir_dir / self.dataset_name)).load(split="test")
        
        # Load filtered corpus
        corpus_file = Path(__file__).parent.parent.parent / "xuyang" / "data" / f"{self.dataset_name}_50" / str(self.corpus_size) / f"corpus_filtered_{self.corpus_size}.csv"
        
        if corpus_file.exists():
            corpus_df = pd.read_csv(corpus_file)
            self.corpus = {row['id']: {'title': row.get('title', ''), 'text': row['text']} for _, row in corpus_df.iterrows()}
        else:
            # Fallback to full corpus
            self.logger.warning(f"Filtered corpus not found at {corpus_file}, using full corpus")
            self.corpus = corpus
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model and return comprehensive metrics"""
        self.logger.info(f"Evaluating model on {self.dataset_name} with {len(self.corpus)} documents")
        
        results = self.evaluator.retrieve(self.corpus, self.queries)
        ndcg, map_score, recall, _ = self.evaluator.evaluate(self.qrels, results, self.evaluator.k_values)
        
        metrics = {}
        for k in [1, 3, 5, 10]:
            metrics[f'ndcg@{k}'] = ndcg[f'NDCG@{k}']
            metrics[f'map@{k}'] = map_score[f'MAP@{k}']
            metrics[f'recall@{k}'] = recall[f'Recall@{k}']
        
        # MRR calculation
        mrr = self.evaluator.evaluate_custom(self.qrels, results, self.evaluator.k_values, metric="mrr")
        metrics['mrr@10'] = mrr['MRR@10']
        
        return metrics
    
    def get_rl_reward(self, metrics: Dict[str, float]) -> float:
        """Compute RL reward from metrics using weighted combination"""
        # Reward weights as defined in config
        reward_weights = {
            'ndcg@10': 0.4,
            'map@10': 0.3,
            'recall@10': 0.2,
            'mrr@10': 0.1
        }
        
        reward = sum(metrics[metric] * weight for metric, weight in reward_weights.items() if metric in metrics)
        return reward
    
    def evaluate_with_reward(self) -> Tuple[Dict[str, float], float]:
        """Evaluate and return both metrics and reward"""
        metrics = self.evaluate()
        reward = self.get_rl_reward(metrics)
        return metrics, reward


class RewardTracker:
    """Tracks reward history for convergence detection"""
    
    def __init__(self, window_size: int = 3):
        self.rewards = []
        self.window_size = window_size
    
    def add_reward(self, reward: float):
        """Add new reward to history"""
        self.rewards.append(reward)
    
    def get_recent_avg(self) -> float:
        """Get average of recent rewards"""
        if len(self.rewards) < self.window_size:
            return np.mean(self.rewards) if self.rewards else 0.0
        return np.mean(self.rewards[-self.window_size:])
    
    def check_convergence(self, threshold: float = 0.001) -> bool:
        """Check if optimization has converged"""
        if len(self.rewards) < self.window_size:
            return False
        
        recent_rewards = self.rewards[-self.window_size:]
        improvement = (recent_rewards[-1] - recent_rewards[0]) / abs(recent_rewards[0])
        
        return abs(improvement) < threshold
    
    def get_best_reward(self) -> float:
        """Get best reward achieved"""
        return max(self.rewards) if self.rewards else 0.0
