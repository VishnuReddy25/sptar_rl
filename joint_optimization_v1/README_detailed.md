# Joint Optimization Framework for SPTAR - Detailed Documentation

This framework implements joint optimization between soft prompt generation and dense retrieval training using reinforcement learning feedback loops.

## Overview

The framework alternates between:
1. **Prompt Optimization**: Using RL to optimize soft prompts based on retrieval performance
2. **Weak Query Generation**: Generating weak supervision data with optimized prompts
3. **DPR Training**: Training dense retrievers with RL-enhanced loss functions
4. **Evaluation**: Computing retrieval metrics to provide feedback for the next iteration

## Architecture & Detailed File Descriptions

```
joint_optimization_v1/
├── core/
│   ├── config.py              # Configuration management - defines JointOptimizationConfig class
│   │                           # with dataset settings, model paths, optimization parameters, and reward weights
│   └── joint_optimizer.py     # Main orchestrator - coordinates the entire optimization loop,
│                           # manages iteration execution, convergence checking, and result saving
├── evaluation/
│   └── retrieval_evaluator.py # Retrieval evaluation and RL rewards - loads BEIR datasets,
│                           # evaluates models using NDCG/MAP/Recall/MRR metrics, computes RL rewards
├── components/
│   ├── prompt_optimizer.py    # RL-based prompt optimization - implements policy gradient updates
│   │                           # for soft prompts based on retrieval feedback, manages prompt parameters
│   └── data_pipeline.py       # Weak query generation and data management - generates synthetic
│                           # queries from documents using optimized prompts, filters and scores queries
├── training/
│   └── dpr_trainer.py         # DPR training with RL feedback - enhanced training loop with
│                           # RL-enhanced loss functions, model checkpointing, progress tracking
└── run_joint_optimization.py  # Main execution script - command-line interface, argument parsing,
                            # experiment setup, result reporting
```

### Detailed File Descriptions

#### Core Module (`core/`)

**`config.py`**:
- **Purpose**: Central configuration management for the entire framework
- **Key Components**:
  - `JointOptimizationConfig` dataclass: Holds all experiment parameters
  - Dataset settings: `dataset_name`, `corpus_size`
  - Model paths: `initial_model_path`, `base_prompt_path`
  - Optimization parameters: `learning_rate`, `rl_weight`, `convergence_threshold`
  - Output directories: `output_dir`, `log_dir`
  - Reward weights: Predefined combinations for RL feedback (NDCG@10: 40%, MAP@10: 30%, etc.)
- **Usage**: Instantiated once at the start of optimization, passed to all components

**`joint_optimizer.py`**:
- **Purpose**: Main orchestrator that coordinates the entire joint optimization process
- **Key Classes**:
  - `JointOptimizer`: Main class that runs the optimization loop
  - `RewardTracker`: Monitors reward history and detects convergence
- **Key Methods**:
  - `run_optimization()`: Main method that executes the alternating loop
  - `_run_single_iteration()`: Executes one complete iteration of prompt→data→training→evaluation
  - `_save_results()`: Saves comprehensive results and summaries
- **Functionality**: Manages iteration execution, convergence checking, exception handling, and result aggregation

#### Evaluation Module (`evaluation/`)

**`retrieval_evaluator.py`**:
- **Purpose**: Handles all retrieval evaluation and RL reward computation
- **Key Classes**:
  - `RetrievalEvaluator`: Main evaluation class
  - `RewardTracker`: Tracks reward history for convergence (also in joint_optimizer.py)
- **Key Methods**:
  - `evaluate()`: Computes standard IR metrics (NDCG@1/3/5/10, MAP, Recall, MRR)
  - `get_rl_reward()`: Converts metrics to RL reward using weighted combination
  - `evaluate_with_reward()`: Combined evaluation and reward computation
- **Data Loading**: Loads BEIR datasets (FiQA, MSMARCO), handles filtered corpus loading
- **Metrics**: Uses BEIR's EvaluateRetrieval for accurate metric computation

#### Components Module (`components/`)

**`prompt_optimizer.py`**:
- **Purpose**: Implements RL-based prompt optimization using policy gradients
- **Key Classes**:
  - `RLPromptOptimizer`: Main optimizer class
  - `PromptManager`: Manages prompt loading and integration with SPTAR
- **Key Methods**:
  - `optimize_prompts()`: Updates prompt parameters based on RL feedback
  - `get_current_prompts()`: Returns current optimized prompts
  - `save_optimized_prompts()`: Saves prompts to disk
- **RL Implementation**: Uses policy gradients to update soft prompt embeddings based on retrieval rewards

**`data_pipeline.py`**:
- **Purpose**: Manages weak supervision data generation and processing
- **Key Classes**:
  - `WeakQueryGenerator`: Generates synthetic queries from documents
  - `TrainingDataManager`: Combines original and weak supervision data
- **Key Methods**:
  - `generate_weak_queries()`: Creates queries using optimized prompts
  - `filter_and_score_queries()`: Filters and quality-scores generated queries
  - `combine_training_data()`: Merges original labeled data with weak data
- **Integration**: Works with existing SPTAR data processing pipelines

#### Training Module (`training/`)

**`dpr_trainer.py`**:
- **Purpose**: Enhanced DPR training with RL feedback integration
- **Key Classes**:
  - `RL_EnhancedLoss`: Loss function combining ranking loss with RL signals
  - `DPRTrainer`: Main training coordinator
  - `ModelManager`: Handles model saving and versioning
- **Key Methods**:
  - `train_with_rl_feedback()`: Main training method with RL enhancement
  - `_prepare_training_samples()`: Converts data to training format
  - `evaluate_training_progress()`: Monitors training metrics
- **RL Integration**: Modifies loss function to include retrieval performance signals

#### Main Script (`run_joint_optimization.py`)

**Purpose**: Command-line interface and experiment orchestration
- **Key Functions**:
  - `parse_args()`: Command-line argument parsing
  - `setup_experiment()`: Creates experiment directory structure
  - `main()`: Main execution function with error handling
- **Features**: Automatic timestamping, configuration saving, result reporting

## Training Commands with Detailed Explanations

### Basic Commands

#### 1. Quick Test Run (1 iteration)
```bash
python run_joint_optimization.py --dataset_name fiqa --corpus_size 500 --max_iterations 1 --num_epochs 1
```
**What it does**: Runs a single iteration of the optimization loop with minimal training (1 epoch) to verify all components work together. Useful for testing the setup without long training times.
**Files accessed**: Loads corpus from `xuyang/data/fiqa_50/500/`, uses default prompts from `xuyang/data/fiqa_50/default_prompt.py`
**Output**: Creates minimal experiment directory with 1 iteration results

#### 2. Standard FiQA Run
```bash
python run_joint_optimization.py --dataset_name fiqa --corpus_size 500 --max_iterations 5
```
**What it does**: Runs the full optimization on FiQA dataset with 500 documents for 5 iterations. Uses default parameters for epochs (3), batch size (16), learning rate (1e-4), and RL weight (0.1).
**Files accessed**: Same as above, plus BEIR FiQA dataset for evaluation
**Output**: Complete experiment with models, prompts, and data for each iteration

### Advanced Commands

#### 3. MSMARCO Dataset with Custom RL Weight
```bash
python run_joint_optimization.py --dataset_name msmarco --corpus_size 500 --max_iterations 5 --rl_weight 0.15
```
**What it does**: Runs optimization on MSMARCO dataset with adjusted RL feedback weight (0.15 instead of default 0.1). MSMARCO often benefits from slightly lower RL weights for stable training.
**Files accessed**: MSMARCO corpus from `xuyang/data/msmarco_50/500/`, MSMARCO BEIR dataset
**Output**: MSMARCO-specific experiment with adjusted RL parameters

#### 4. High-Resource Training Configuration
```bash
python run_joint_optimization.py --dataset_name fiqa --corpus_size 500 --max_iterations 15 --num_epochs 10 --batch_size 64 --learning_rate 5e-5
```
**What it does**: Extended training with more iterations (15), longer training per iteration (10 epochs), larger batch size (64) for training stability, and lower learning rate (5e-5) for finer prompt optimization.
**Files accessed**: Standard FiQA files with extended processing
**Output**: Large experiment with many iterations and detailed training logs

#### 5. Custom Experiment with Named Output
```bash
python run_joint_optimization.py --dataset_name fiqa --corpus_size 500 --max_iterations 10 --num_epochs 5 --batch_size 32 --learning_rate 1e-4 --rl_weight 0.2 --output_dir custom_experiments --experiment_name fiqa_500_rl_0.2_v1
```
**What it does**: Full control over all parameters with custom output directory and experiment naming. Higher RL weight (0.2) puts more emphasis on retrieval performance feedback.
**Files accessed**: All standard files plus custom output location
**Output**: Experiment saved to `custom_experiments/fiqa_500_rl_0.2_v1/`

### Debugging and Development Commands

#### 6. Debug Mode with Detailed Logging
```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from joint_optimization_v1.run_joint_optimization import main
import sys
sys.argv = ['run_joint_optimization.py', '--dataset_name', 'fiqa', '--corpus_size', '500', '--max_iterations', '1']
main()
"
```
**What it does**: Enables debug-level logging to see detailed information about each step of the optimization process, including intermediate values, file operations, and potential issues.
**Files accessed**: All standard files with verbose logging output
**Output**: Detailed logs showing internal state and operations

#### 7. Memory-Efficient Small Batch Training
```bash
python run_joint_optimization.py --dataset_name fiqa --corpus_size 500 --max_iterations 3 --batch_size 8 --num_epochs 2
```
**What it does**: Uses smaller batch size (8) and fewer epochs (2) for memory-constrained environments. Reduces GPU memory usage while still allowing the optimization loop to function.
**Files accessed**: Standard files with reduced memory footprint
**Output**: Smaller experiment suitable for limited hardware

## Configuration

### Key Parameters

- `dataset_name`: Dataset to use (`fiqa` or `msmarco`)
- `corpus_size`: Size of filtered corpus (500 for fiqa_50/500)
- `max_iterations`: Maximum optimization iterations
- `rl_weight`: Weight for RL feedback in loss functions
- `learning_rate`: Learning rate for prompt optimization
- `convergence_threshold`: Threshold for early stopping

### Output Structure

```
output_dir/
├── config.json                 # Experiment configuration
├── results.json                # Complete optimization results
├── summary.txt                 # Human-readable summary
├── models/                     # Trained DPR models per iteration
├── prompts/                    # Optimized prompts per iteration
├── weak_queries/               # Generated weak queries per iteration
├── training_data/              # Combined training data per iteration
└── logs/                       # Training logs
```

## Integration with Existing SPTAR

This framework integrates with the existing SPTAR codebase:

- Uses `xuyang/` for prompt management and weak query generation
- Uses `zhiyuan/` for DPR training and evaluation components
- Leverages existing data processing and filtering pipelines

## RL Feedback Loop

The optimization follows this cycle:

1. **Evaluate** current retriever on test set
2. **Compute** RL reward from retrieval metrics (NDCG, MAP, MRR)
3. **Optimize** prompts using policy gradients based on reward
4. **Generate** weak queries with optimized prompts
5. **Train** DPR with RL-enhanced loss incorporating retrieval feedback
6. **Repeat** until convergence or max iterations

## Metrics and Rewards

The RL reward is computed as a weighted combination:
- NDCG@10: 40%
- MAP@10: 30%
- Recall@10: 20%
- MRR@10: 10%

## Requirements

- Python 3.7+
- PyTorch
- Sentence Transformers
- BEIR
- Existing SPTAR dependencies

## Example Results

After running on FiQA with 500 documents:

```
Total Iterations: 5
Best Reward: 0.7234
Best Model: iteration_3
Total Time: 1247.32 seconds
Converged at iteration: 4
```

## Troubleshooting

### Common Issues

1. **Missing corpus files**: Ensure filtered corpus exists in `xuyang/data/{dataset}_50/{corpus_size}/`
2. **Model loading errors**: Check that initial model path is valid
3. **Memory issues**: Reduce batch size or use smaller models
4. **Convergence issues**: Adjust RL weight or learning rate

### Debugging

Enable detailed logging:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m joint_optimization_v1.run_joint_optimization --debug
```

## Future Extensions

- Multi-objective optimization (quality vs. diversity)
- Curriculum learning for progressive corpus sizes
- Meta-learning across different datasets
- Integration with other retrievers (ColBERT, DPR variants)
