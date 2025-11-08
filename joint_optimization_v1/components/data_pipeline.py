"""
Data Pipeline for Weak Query Generation

Handles the generation and filtering of weak supervision data using optimized prompts.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))


class WeakQueryGenerator:
    """Generates weak queries using optimized prompts"""
    
    def __init__(self, dataset_name: str = "fiqa", corpus_size: int = 500, prompt_manager=None):
        self.dataset_name = dataset_name
        self.corpus_size = corpus_size
        self.prompt_manager = prompt_manager
        
        # Data paths
        self.data_dir = Path(__file__).parent.parent.parent / "xuyang" / "data" / f"{dataset_name}_50" / str(corpus_size)
        
        # Load corpus
        self.corpus = self._load_corpus()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_corpus(self) -> Dict[str, Dict]:
        """Load filtered corpus"""
        corpus_file = self.data_dir / f"corpus_filtered_{self.corpus_size}.csv"
        
        if corpus_file.exists():
            corpus_df = pd.read_csv(corpus_file)
            corpus = {}
            for _, row in corpus_df.iterrows():
                corpus[str(row['id'])] = {
                    'title': row.get('title', ''),
                    'text': row['text']
                }
            return corpus
        else:
            self.logger.error(f"Corpus file not found: {corpus_file}")
            return {}
    
    def generate_weak_queries(self, prompts: Dict, num_queries_per_doc: int = 5) -> List[Dict]:
        """Generate weak queries using provided prompts"""
        self.logger.info(f"Generating weak queries for {len(self.corpus)} documents")
        
        weak_queries = []
        
        for doc_id, doc_data in self.corpus.items():
            document_text = doc_data.get('text', '')
            
            # Generate queries for each prompt
            for prompt_key, prompt_data in prompts.items():
                if 'template' in prompt_data:
                    template = prompt_data['template']
                    
                    # Generate multiple queries per document-prompt pair
                    for i in range(num_queries_per_doc):
                        query = self._generate_single_query(template, document_text, doc_id, i)
                        if query:
                            weak_queries.append({
                                'query': query,
                                'doc_id': doc_id,
                                'prompt_key': prompt_key,
                                'iteration': prompt_data.get('iteration', 0),
                                'timestamp': datetime.now().isoformat()
                            })
        
        self.logger.info(f"Generated {len(weak_queries)} weak query candidates")
        return weak_queries
    
    def _generate_single_query(self, template: str, document: str, doc_id: str, attempt: int) -> Optional[str]:
        """Generate a single query (simplified - in practice would call LLM)"""
        # This is a placeholder - in the real implementation, this would call
        # the actual LLM inference from weak_inference.py
        
        # For now, create synthetic queries based on document content
        if len(document) < 50:
            return None
            
        # Extract key phrases (simplified)
        words = document.split()[:10]  # First 10 words
        query = f"What is {' '.join(words)}?"
        
        return query
    
    def filter_and_score_queries(self, weak_queries: List[Dict]) -> Tuple[List[Dict], float]:
        """Filter weak queries and compute quality score"""
        # This would integrate with the existing filtering pipeline
        # For now, simple filtering based on length and uniqueness
        
        filtered_queries = []
        seen_queries = set()
        
        for query_data in weak_queries:
            query = query_data['query']
            
            # Basic filtering
            if len(query.split()) < 3:  # Too short
                continue
            if len(query) > 200:  # Too long
                continue
            if query.lower() in seen_queries:  # Duplicate
                continue
            
            seen_queries.add(query.lower())
            filtered_queries.append(query_data)
        
        # Quality score (simplified)
        quality_score = len(filtered_queries) / len(weak_queries) if weak_queries else 0.0
        
        self.logger.info(f"Filtered {len(weak_queries)} -> {len(filtered_queries)} queries (quality: {quality_score:.3f})")
        
        return filtered_queries, quality_score
    
    def save_weak_queries(self, queries: List[Dict], output_path: str):
        """Save weak queries to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for query in queries:
                f.write(json.dumps(query) + '\n')
        
        self.logger.info(f"Saved {len(queries)} weak queries to {output_path}")


class TrainingDataManager:
    """Manages training data composition for DPR training"""
    
    def __init__(self, dataset_name: str = "fiqa"):
        self.dataset_name = dataset_name
        self.data_dir = Path(__file__).parent.parent.parent / "xuyang" / "data" / f"{dataset_name}_50"
        
        # Load original labeled data
        self.original_data = self._load_original_data()
    
    def _load_original_data(self) -> pd.DataFrame:
        """Load original labeled training data"""
        train_file = self.data_dir / "prompt_tuning_50_train.tsv"
        
        if train_file.exists():
            return pd.read_csv(train_file, sep='\t')
        else:
            self.logger.warning(f"Original training data not found: {train_file}")
            return pd.DataFrame()
    
    def combine_training_data(self, weak_queries: List[Dict], 
                            weak_to_original_ratio: float = 1.0) -> pd.DataFrame:
        """Combine original and weak supervision data"""
        
        # Convert weak queries to DataFrame format
        weak_df = pd.DataFrame(weak_queries)
        
        # For simplicity, create a basic format
        # In practice, this would match the expected training format
        
        combined_data = []
        
        # Add original data
        for _, row in self.original_data.iterrows():
            combined_data.append({
                'query': row.get('query', ''),
                'positive': row.get('positive', ''),
                'negatives': row.get('negatives', ''),
                'source': 'original'
            })
        
        # Add weak data (sample to match ratio)
        num_weak = int(len(self.original_data) * weak_to_original_ratio)
        weak_sample = weak_df.sample(min(num_weak, len(weak_df)))
        
        for _, row in weak_sample.iterrows():
            combined_data.append({
                'query': row['query'],
                'positive': row['doc_id'],
                'negatives': '',  # Would need to be generated
                'source': 'weak'
            })
        
        return pd.DataFrame(combined_data)
    
    def save_training_data(self, data: pd.DataFrame, output_path: str):
        """Save combined training data"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_path, index=False)
        logging.getLogger(__name__).info(f"Saved training data to {output_path}")
