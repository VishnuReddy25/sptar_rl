"""
Joint Optimization Orchestrator

Main controller that coordinates the alternating training between prompts and retrievers.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

from core.config import JointOptimizationConfig
from evaluation.retrieval_evaluator import RetrievalEvaluator, RewardTracker
from components.prompt_optimizer import RLPromptOptimizer, PromptManager
from components.data_pipeline import WeakQueryGenerator, TrainingDataManager
from training.dpr_trainer import DPRTrainer, ModelManager


class JointOptimizer:
    """Main orchestrator for joint prompt-retriever optimization"""
    
    def __init__(self, config: JointOptimizationConfig):
        self.config = config
        self.current_iteration = 0
        
        # Initialize components
        self.evaluator = RetrievalEvaluator(
            model_path=config.initial_model_path,
            dataset_name=config.dataset_name,
            corpus_size=config.corpus_size
        )
        
        self.reward_tracker = RewardTracker()
        
        self.prompt_optimizer = RLPromptOptimizer(
            base_prompt_path=config.base_prompt_path,
            learning_rate=config.learning_rate,
            reward_weight=config.rl_weight
        )
        
        self.prompt_manager = PromptManager(config.base_prompt_path)
        
        self.data_generator = WeakQueryGenerator(
            dataset_name=config.dataset_name,
            corpus_size=config.corpus_size,
            prompt_manager=self.prompt_manager
        )
        
        self.training_manager = TrainingDataManager(config.dataset_name)
        
        self.dpr_trainer = DPRTrainer(
            base_model_path=config.initial_model_path,
            output_dir=str(config.output_dir / "models"),
            rl_weight=config.rl_weight
        )
        
        self.model_manager = ModelManager(str(config.output_dir / "models"))
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Joint Optimizer initialized")
    
    def _setup_logging(self):
        """Setup logging for the optimization process"""
        log_file = self.config.log_dir / f"joint_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_optimization(self) -> Dict:
        """Run the complete joint optimization loop"""
        self.logger.info("Starting joint optimization")
        start_time = time.time()
        
        results = {
            'iterations': [],
            'best_model': None,
            'best_reward': 0.0,
            'convergence_iteration': None,
            'total_time': 0.0
        }
        
        try:
            for iteration in range(self.config.max_iterations):
                self.current_iteration = iteration
                self.logger.info(f"Starting iteration {iteration}")
                
                iteration_start = time.time()
                iteration_result = self._run_single_iteration()
                iteration_time = time.time() - iteration_start
                
                results['iterations'].append({
                    **iteration_result,
                    'iteration_time': iteration_time
                })
                
                # Update best model
                if iteration_result['reward'] > results['best_reward']:
                    results['best_reward'] = iteration_result['reward']
                    results['best_model'] = iteration_result['model_path']
                
                # Check convergence
                self.reward_tracker.add_reward(iteration_result['reward'])
                if self.reward_tracker.check_convergence(self.config.convergence_threshold):
                    self.logger.info(f"Convergence detected at iteration {iteration}")
                    results['convergence_iteration'] = iteration
                    break
            
            results['total_time'] = time.time() - start_time
            self._save_results(results)
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
        
        self.logger.info("Joint optimization completed")
        return results
    
    def _run_single_iteration(self) -> Dict:
        """Run a single iteration of the optimization loop"""
        iteration_results = {
            'iteration': self.current_iteration,
            'reward': 0.0,
            'metrics': {},
            'model_path': None,
            'weak_queries_generated': 0,
            'training_samples': 0
        }
        
        try:
            # Step 1: Evaluate current retriever
            self.logger.info("Step 1: Evaluating current retriever")
            metrics, reward = self.evaluator.evaluate_with_reward()
            iteration_results['metrics'] = metrics
            iteration_results['reward'] = reward
            
            # Step 2: Optimize prompts based on reward
            self.logger.info("Step 2: Optimizing prompts")
            optimized_prompts = self.prompt_optimizer.optimize_prompts(reward)
            self.prompt_manager.update_prompts(optimized_prompts)
            
            # Save optimized prompts
            prompt_path = self.config.output_dir / "prompts" / f"prompts_iteration_{self.current_iteration}.json"
            self.prompt_optimizer.save_optimized_prompts(str(prompt_path), optimized_prompts)
            
            # Step 3: Generate weak queries with optimized prompts
            self.logger.info("Step 3: Generating weak queries")
            weak_queries = self.data_generator.generate_weak_queries(optimized_prompts)
            filtered_queries, quality_score = self.data_generator.filter_and_score_queries(weak_queries)
            
            # Save weak queries
            weak_query_path = self.config.output_dir / "weak_queries" / f"weak_queries_iteration_{self.current_iteration}.jsonl"
            self.data_generator.save_weak_queries(filtered_queries, str(weak_query_path))
            
            iteration_results['weak_queries_generated'] = len(filtered_queries)
            
            # Step 4: Prepare training data
            self.logger.info("Step 4: Preparing training data")
            training_data = self.training_manager.combine_training_data(filtered_queries)
            iteration_results['training_samples'] = len(training_data)
            
            # Save training data
            train_data_path = self.config.output_dir / "training_data" / f"train_data_iteration_{self.current_iteration}.csv"
            self.training_manager.save_training_data(training_data, str(train_data_path))
            
            # Step 5: Train DPR with RL feedback
            self.logger.info("Step 5: Training DPR model")
            model_path = self.dpr_trainer.train_with_rl_feedback(
                training_data=training_data,
                num_epochs=self.config.num_epochs,
                batch_size=self.config.batch_size,
                iteration=self.current_iteration
            )
            
            # Save model
            saved_model_path = self.model_manager.save_iteration_model(model_path, self.current_iteration)
            iteration_results['model_path'] = saved_model_path
            
            # Update evaluator with new model
            self.evaluator = RetrievalEvaluator(
                model_path=saved_model_path,
                dataset_name=self.config.dataset_name,
                corpus_size=self.config.corpus_size
            )
            
            self.logger.info(f"Iteration {self.current_iteration} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Iteration {self.current_iteration} failed: {e}")
            raise
        
        return iteration_results
    
    def _save_results(self, results: Dict):
        """Save optimization results"""
        results_path = self.config.output_dir / "results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_path}")
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of the optimization process"""
        return {
            'total_iterations': len(self.reward_tracker.rewards),
            'best_reward': self.reward_tracker.get_best_reward(),
            'reward_history': self.reward_tracker.rewards,
            'reward_stats': self.prompt_optimizer.get_reward_stats(),
            'config': {
                'dataset': self.config.dataset_name,
                'corpus_size': self.config.corpus_size,
                'max_iterations': self.config.max_iterations,
                'rl_weight': self.config.rl_weight
            }
        }


def run_joint_optimization(config: JointOptimizationConfig) -> Dict:
    """Convenience function to run joint optimization"""
    optimizer = JointOptimizer(config)
    return optimizer.run_optimization()
