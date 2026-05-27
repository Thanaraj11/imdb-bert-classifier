"""Inference pipeline for sentiment prediction"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.model_config import CONFIG
from typing import Dict, List

class SentimentPredictor:
    """Handles model inference for single/batch predictions"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor with saved model.
        
        Args:
            model_path: Path to saved model directory
        """
        self.model_path = model_path or CONFIG.output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = {0: "negative", 1: "positive"}
        self.max_length = CONFIG.max_length
        
        print(f"✅ Predictor ready on {self.device}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single review.
        
        Args:
            text: Movie review text
            
        Returns:
            Dictionary with 'label' and 'confidence'
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Post-process
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        
        return {
            "label": self.id2label[pred_id],
            "confidence": round(probs[pred_id].item(), 4),
            "raw_probabilities": {
                "negative": round(probs[0].item(), 4),
                "positive": round(probs[1].item(), 4)
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for multiple reviews.
        
        Args:
            texts: List of review texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def get_confidence_interval(self, text: str, num_samples: int = 10) -> Dict:
        """Monte Carlo dropout for uncertainty estimation"""
        self.model.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            
            for _ in range(num_samples):
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                predictions.append(probs[1].item())  # Positive class probability
        
        self.model.eval()
        
        return {
            "mean_confidence": np.mean(predictions),
            "std_confidence": np.std(predictions),
            "lower_bound": np.percentile(predictions, 2.5),
            "upper_bound": np.percentile(predictions, 97.5)
        }