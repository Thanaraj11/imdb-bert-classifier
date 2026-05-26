"""Model configuration and hyperparameters"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """BERT sentiment classifier configuration"""
    
    # Model architecture
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    max_length: int = 256
    
    # Training hyperparameters
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 2
    weight_decay: float = 0.01
    warmup_steps: int = 500
    fp16: bool = True  # Enable if GPU available
    
    # Data configuration
    train_size: int = 2000
    test_size: int = 500
    validation_split: float = 0.1
    
    # Paths
    output_dir: str = "./saved_models/imdb_bert"
    cache_dir: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        self.fp16 = self.fp16 and self._has_cuda()
    
    @staticmethod
    def _has_cuda() -> bool:
        import torch
        return torch.cuda.is_available()

# Singleton instance
CONFIG = ModelConfig()