"""BERT model initialization and setup"""

from transformers import AutoModelForSequenceClassification
from config.model_config import CONFIG
import torch

def initialize_model():
    """
    Load pre-trained BERT with classification head.
    
    Returns:
        Model instance ready for fine-tuning
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG.model_name,
        num_labels=CONFIG.num_labels,
        cache_dir=CONFIG.cache_dir
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ Loaded {CONFIG.model_name} on {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device

def get_model_info(model) -> dict:
    """Return model architecture summary"""
    return {
        "model_name": CONFIG.model_name,
        "num_labels": CONFIG.num_labels,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": next(model.parameters()).device.type
    }