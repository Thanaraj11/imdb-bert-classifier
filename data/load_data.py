"""IMDB dataset loading and preparation"""

from datasets import load_dataset, DatasetDict
from typing import Tuple
from config.model_config import CONFIG
import numpy as np

def load_imdb_data() -> DatasetDict:
    """
    Load the IMDB dataset from Hugging Face.
    
    Returns:
        DatasetDict containing 'train' and 'test' splits
    """
    dataset = load_dataset("imdb", cache_dir=CONFIG.cache_dir)
    print(f"✅ Loaded IMDB dataset: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    return dataset

def create_small_subset(dataset: DatasetDict) -> DatasetDict:
    """
    Create smaller subsets for quick experimentation.
    
    Args:
        dataset: Full IMDB dataset
        
    Returns:
        DatasetDict with reduced train/test splits
    """
    small_train = dataset["train"].shuffle(seed=CONFIG.seed).select(range(CONFIG.train_size))
    small_test = dataset["test"].shuffle(seed=CONFIG.seed).select(range(CONFIG.test_size))
    
    return DatasetDict({
        "train": small_train,
        "test": small_test
    })

def train_val_split(dataset: DatasetDict) -> Tuple:
    """
    Split training data into train and validation sets.
    
    Args:
        dataset: DatasetDict with 'train' split
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    split = dataset["train"].train_test_split(
        test_size=CONFIG.validation_split, 
        seed=CONFIG.seed
    )
    return split["train"], split["test"]