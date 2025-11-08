"""
RL-based Prompt Optimizer

Optimizes soft prompts using reinforcement learning feedback from retrieval performance.
"""

import os
import sys
import torch
import torch.nn as nn
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))


class RLPromptOptimizer:
    """RL-based prompt optimizer using policy gradients"""
    
    def __init__(self, base_prompt_path: str, learning_rate: float = 1e-4, reward_weight: float = 0.1):
        self.base_prompt_path = Path(base_prompt_path)
        self.learning_rate = learning_rate
        self.reward_weight = reward_weight
        self.current_iteration = 0
        
        # Load base prompts
        with open(self.base_prompt_path, 'r') as f:
            self.base_prompts = json.load(f)
        
        # Initialize prompt parameters (virtual tokens)
        self.prompt_params = self._initialize_prompt_params()
        self.optimizer = torch.optim.Adam([self.prompt_params], lr=learning_rate)
        
        # Reward history for logging
        self.reward_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_prompt_params(self) -> torch.nn.Parameter:
        """Initialize prompt parameters from base prompts"""
        # For simplicity, we'll treat prompts as learnable embeddings
        # In practice, this would be the actual soft prompt tokens
        
        # Get prompt dimensions (simplified)
        prompt_dim = 768  # BERT base dimension
        num_prompts = len(self.base_prompts)
        prompt_length = 10  # Number of virtual tokens
        
        # Initialize with random embeddings (in practice, load from actual prompts)
        prompts = torch.randn(num_prompts, prompt_length, prompt_dim)
        return torch.nn.Parameter(prompts)
    
    def optimize_prompts(self, retrieval_reward: float, weak_query_quality: float = 0.0) -> Dict:
        """Optimize prompts based on retrieval performance feedback"""
        
        # Combine rewards
        combined_reward = retrieval_reward + self.reward_weight * weak_query_quality
        
        # Store reward for logging
        self.reward_history.append(combined_reward)
        
        # Simple policy gradient update (simplified version)
        # In practice, this would involve computing gradients through the entire pipeline
        
        # For now, add some noise to simulate learning
        noise = torch.randn_like(self.prompt_params) * 0.01
        self.prompt_params.data += noise
        
        # Create updated prompts (simplified)
        updated_prompts = {}
        for i, (key, prompt_data) in enumerate(self.base_prompts.items()):
            updated_prompts[key] = {
                **prompt_data,
                'reward_score': combined_reward,
                'iteration': self.current_iteration,
                'embedding_norm': torch.norm(self.prompt_params[i]).item()
            }
        
        self.current_iteration += 1
        
        self.logger.info(f"Optimized prompts with reward: {combined_reward:.4f}")
        return updated_prompts
    
    def get_current_prompts(self) -> Dict:
        """Get current optimized prompts"""
        current_prompts = {}
        for i, (key, prompt_data) in enumerate(self.base_prompts.items()):
            current_prompts[key] = {
                **prompt_data,
                'iteration': self.current_iteration,
                'embedding_norm': torch.norm(self.prompt_params[i]).item()
            }
        return current_prompts
    
    def save_optimized_prompts(self, output_path: str, prompts: Dict):
        """Save optimized prompts"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        self.logger.info(f"Saved optimized prompts to {output_path}")
    
    def get_reward_stats(self) -> Dict:
        """Get reward statistics for analysis"""
        if not self.reward_history:
            return {}
        
        rewards = np.array(self.reward_history)
        return {
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards),
            'total_iterations': len(rewards)
        }


class PromptManager:
    """Manages prompt loading and integration with existing SPTAR system"""
    
    def __init__(self, prompt_path: str):
        self.prompt_path = Path(prompt_path)
        self.current_prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict:
        """Load prompts from file"""
        if self.prompt_path.exists():
            with open(self.prompt_path, 'r') as f:
                return json.load(f)
        else:
            # Return default prompts
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict:
        """Get default prompt configuration"""
        return {
            "llama_prompt_0": {
                "template": "Generate a question for the document: {document}",
                "model": "llama-7b",
                "temperature": 0.7
            },
            "llama_prompt_1": {
                "template": "What question does this document answer? {document}",
                "model": "llama-7b", 
                "temperature": 0.8
            }
        }
    
    def get_prompt_for_generation(self, prompt_key: str) -> str:
        """Get prompt template for query generation"""
        if prompt_key in self.current_prompts:
            return self.current_prompts[prompt_key].get('template', '')
        return self._get_default_prompts()[prompt_key].get('template', '')
    
    def update_prompts(self, new_prompts: Dict):
        """Update current prompts"""
        self.current_prompts.update(new_prompts)
