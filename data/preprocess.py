"""Tokenization and preprocessing for BERT"""

from transformers import AutoTokenizer
from datasets import Dataset
from config.model_config import CONFIG
from typing import Dict

class BERTPreprocessor:
    """Handles tokenization for BERT-based models"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
        self.max_length = CONFIG.max_length
        
    def tokenize_function(self, batch: Dict) -> Dict:
        """
        Tokenize a batch of text reviews.
        
        Args:
            batch: Dictionary containing 'text' field
            
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding=False,  # Will pad dynamically with DataCollator
            max_length=self.max_length
        )
    
    def preprocess_datasets(self, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> tuple:
        """
        Tokenize all three datasets.
        
        Args:
            train_ds, val_ds, test_ds: Raw datasets
            
        Returns:
            Tokenized versions of each dataset
        """
        train_tokenized = train_ds.map(self.tokenize_function, batched=True)
        val_tokenized = val_ds.map(self.tokenize_function, batched=True)
        test_tokenized = test_ds.map(self.tokenize_function, batched=True)
        
        print(f"✅ Tokenized datasets: {len(train_tokenized)} train, {len(val_tokenized)} val, {len(test_tokenized)} test")
        
        return train_tokenized, val_tokenized, test_tokenized
    
    def get_tokenizer(self):
        """Return the tokenizer for inference"""
        return self.tokenizer