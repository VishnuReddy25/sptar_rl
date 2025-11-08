"""
Configuration for Joint Optimization Framework
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class JointOptimizationConfig:
    """Configuration for the joint optimization framework"""
    
    # Dataset settings
    dataset_name: str = "fiqa"
    corpus_size: int = 500
    
    # Model paths
    initial_model_path: str = "zhiyuan/retriever/dpr/train/output/llama_7b_500_fixed_v3_best_llama_prompt_2_filtered_70_filtered_50/50/bert-base-uncased-v1-fiqa"
    base_prompt_path: str = "xuyang/data/fiqa_50/default_prompt.py"
    
    # Optimization settings
    max_iterations: int = 5
    convergence_threshold: float = 0.001
    
    # RL settings
    rl_weight: float = 0.1
    learning_rate: float = 1e-4
    
    # Training settings
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    
    # Evaluation settings
    eval_batch_size: int = 32
    k_values: List[int] = None
    
    # Paths
    output_dir: str = "joint_optimization_v1/output"
    log_dir: str = "joint_optimization_v1/logs"
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10, 100]
        
        # Convert string paths to Path objects
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Reward function weights
REWARD_WEIGHTS = {
    'ndcg@10': 0.4,
    'map@10': 0.3,
    'recall@10': 0.2,
    'mrr@10': 0.1
}

# Default configurations for different datasets
DATASET_CONFIGS = {
    'fiqa': JointOptimizationConfig(
        dataset_name='fiqa',
        corpus_size=500,
        initial_model_path="zhiyuan/retriever/dpr/train/output/llama_7b_500_fixed_v3_best_llama_prompt_2_filtered_70_filtered_50/50/bert-base-uncased-v1-fiqa",
        base_prompt_path="xuyang/data/fiqa_50/default_prompt.py"
    ),
    'msmarco': JointOptimizationConfig(
        dataset_name='msmarco',
        corpus_size=1000,
        initial_model_path="zhiyuan/retriever/dpr/train/output/msmarco_baseline",
        base_prompt_path="xuyang/data/msmarco_50/default_prompt.py"
    )
}
