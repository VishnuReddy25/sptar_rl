#!/usr/bin/env python3
"""
Main Script for Joint Optimization

Runs the complete joint optimization framework for SPTAR with RL feedback.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))

from core.config import JointOptimizationConfig
from core.joint_optimizer import run_joint_optimization


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Joint Optimization for SPTAR")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="fiqa", 
                       choices=["fiqa", "msmarco"], help="Dataset name")
    parser.add_argument("--corpus_size", type=int, default=500, 
                       help="Size of filtered corpus")
    
    # Model paths
    parser.add_argument("--initial_model_path", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Initial model path")
    parser.add_argument("--base_prompt_path", type=str,
                       default="xuyang/data/fiqa_50/default_prompt.py",
                       help="Base prompt configuration path")
    
    # Optimization parameters
    parser.add_argument("--max_iterations", type=int, default=5,
                       help="Maximum optimization iterations")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs per DPR training")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for prompt optimization")
    parser.add_argument("--rl_weight", type=float, default=0.1,
                       help="Weight for RL feedback")
    parser.add_argument("--convergence_threshold", type=float, default=0.001,
                       help="Convergence threshold")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="joint_optimization_output",
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name (auto-generated if not provided)")
    
    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment directory and configuration"""
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{args.dataset_name}_{args.corpus_size}_{timestamp}"
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "prompts").mkdir(exist_ok=True)
    (output_dir / "weak_queries").mkdir(exist_ok=True)
    (output_dir / "training_data").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # Create configuration
    config = JointOptimizationConfig(
        dataset_name=args.dataset_name,
        corpus_size=args.corpus_size,
        initial_model_path=args.initial_model_path,
        base_prompt_path=args.base_prompt_path,
        max_iterations=args.max_iterations,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rl_weight=args.rl_weight,
        convergence_threshold=args.convergence_threshold,
        output_dir=output_dir,
        log_dir=output_dir / "logs"
    )
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return config


def main():
    """Main execution function"""
    args = parse_args()
    
    print("=== Joint Optimization for SPTAR ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Corpus Size: {args.corpus_size}")
    print(f"Max Iterations: {args.max_iterations}")
    print("=" * 40)
    
    try:
        # Setup experiment
        config = setup_experiment(args)
        print(f"Experiment directory: {config.output_dir}")
        
        # Run optimization
        print("Starting joint optimization...")
        results = run_joint_optimization(config)
        
        # Print summary
        print("\n=== Optimization Complete ===")
        print(f"Total Iterations: {len(results['iterations'])}")
        print(f"Best Reward: {results['best_reward']:.4f}")
        print(f"Best Model: {results['best_model']}")
        print(f"Total Time: {results['total_time']:.2f} seconds")
        
        if results['convergence_iteration'] is not None:
            print(f"Converged at iteration: {results['convergence_iteration']}")
        
        # Save final summary
        summary_path = config.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Joint Optimization Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Corpus Size: {args.corpus_size}\n")
            f.write(f"Total Iterations: {len(results['iterations'])}\n")
            f.write(f"Best Reward: {results['best_reward']:.4f}\n")
            f.write(f"Total Time: {results['total_time']:.2f} seconds\n")
            f.write(f"Best Model: {results['best_model']}\n")
        
        print(f"Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
