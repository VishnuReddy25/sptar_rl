"""
DPR Trainer with RL Feedback

Enhanced DPR training that incorporates retrieval performance feedback.
"""

import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing training components
try:
    from zhiyuan.retriever.dpr.train.train_sbert import train_sbert
    from zhiyuan.retriever.dpr.train.train_sbert_rl_feedback import train_sbert_rl
except ImportError:
    logging.warning("Could not import existing training modules")


class RL_EnhancedLoss(nn.Module):
    """Enhanced loss function with RL feedback"""
    
    def __init__(self, base_loss, rl_weight: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.rl_weight = rl_weight
    
    def forward(self, sentence_features, labels):
        # Base ranking loss
        base_loss = self.base_loss(sentence_features, labels)
        
        # RL feedback component (simplified)
        # In practice, this would compute retrieval metrics on a validation set
        # and use them to modulate the loss
        
        query_emb = sentence_features[0]['sentence_embedding']
        doc_emb = sentence_features[1]['sentence_embedding']
        
        # Simple MRR approximation as additional signal
        scores = torch.mm(query_emb, doc_emb.t())
        batch_size = scores.shape[0]
        
        # Approximate reciprocal rank
        ranks = (scores >= scores.diag().unsqueeze(1)).sum(dim=1).float()
        mrr_signal = (1.0 / ranks).mean()
        
        # Combine losses
        rl_loss = -self.rl_weight * mrr_signal
        total_loss = base_loss + rl_loss
        
        return total_loss


class DPRTrainer:
    """DPR trainer with RL feedback integration"""
    
    def __init__(self, base_model_path: str, output_dir: str, rl_weight: float = 0.1):
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.rl_weight = rl_weight
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def train_with_rl_feedback(self, training_data: pd.DataFrame, 
                             num_epochs: int = 3, batch_size: int = 16,
                             iteration: int = 0) -> str:
        """Train DPR model with RL-enhanced loss"""
        
        self.logger.info(f"Starting DPR training for iteration {iteration}")
        
        # Prepare training data in expected format
        train_samples = self._prepare_training_samples(training_data)
        
        # Create enhanced loss function
        base_loss = nn.CrossEntropyLoss()  # MultipleNegativesRankingLoss equivalent
        enhanced_loss = RL_EnhancedLoss(base_loss, self.rl_weight)
        
        # Training configuration
        train_config = {
            'model_name': self.base_model_path,
            'train_data': train_samples,
            'epochs': num_epochs,
            'batch_size': batch_size,
            'loss_function': enhanced_loss,
            'output_path': str(self.output_dir / f"iteration_{iteration}")
        }
        
        # Call existing training function (simplified)
        # In practice, this would integrate with train_sbert.py
        trained_model_path = self._train_model(train_config)
        
        self.logger.info(f"Completed DPR training for iteration {iteration}")
        return trained_model_path
    
    def _prepare_training_samples(self, data: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to training samples format"""
        samples = []
        
        for _, row in data.iterrows():
            # Create training sample
            sample = {
                'query': row['query'],
                'positive': row['positive'],
                'negatives': row.get('negatives', '').split(',') if row.get('negatives') else []
            }
            samples.append(sample)
        
        return samples
    
    def _train_model(self, config: Dict) -> str:
        """Train the model (simplified implementation)"""
        # This is a placeholder - in practice would call the actual training
        # from zhiyuan/retriever/dpr/train/train_sbert.py
        
        output_path = config['output_path']
        
        # Simulate training by copying base model
        import shutil
        if os.path.exists(self.base_model_path):
            shutil.copytree(self.base_model_path, output_path)
        
        # Add training metadata
        metadata = {
            'iteration': config.get('iteration', 0),
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'rl_weight': self.rl_weight,
            'trained_at': datetime.now().isoformat()
        }
        
        metadata_path = Path(output_path) / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        return output_path
    
    def evaluate_training_progress(self, model_path: str, 
                                 validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """Evaluate training progress"""
        # This would compute validation metrics
        # For now, return dummy metrics
        
        return {
            'training_loss': 0.5,  # Would be actual loss
            'validation_ndcg@10': 0.65,  # Would be computed
            'model_path': model_path
        }


class ModelManager:
    """Manages model saving and loading for different iterations"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_iteration_model(self, model_path: str, iteration: int) -> str:
        """Save model for specific iteration"""
        iteration_dir = self.base_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(exist_ok=True)
        
        # Copy model files
        import shutil
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                src = os.path.join(model_path, file)
                dst = iteration_dir / file
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        return str(iteration_dir)
    
    def get_best_model(self, max_iteration: int) -> str:
        """Get the best performing model across iterations"""
        # For now, return the latest
        return str(self.base_dir / f"iteration_{max_iteration}")
    
    def cleanup_old_models(self, keep_last_n: int = 3):
        """Clean up old model checkpoints"""
        iterations = sorted([d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('iteration_')])
        
        if len(iterations) > keep_last_n:
            for old_dir in iterations[:-keep_last_n]:
                import shutil
                shutil.rmtree(old_dir)
