"""End-to-end training pipeline"""

import sys
sys.path.append('.')

from data.load_data import load_imdb_data, create_small_subset, train_val_split
from data.preprocess import BERTPreprocessor
from models.train import train_model, evaluate_model
from evaluation.predict import SentimentPredictor
from config.model_config import CONFIG
import torch

def main():
    """Run complete training pipeline"""
    print("="*50)
    print("IMDB Sentiment Classifier - Training Pipeline")
    print("="*50)
    
    # 1. Load data
    print("\n[1/6] Loading IMDB dataset...")
    full_dataset = load_imdb_data()
    small_dataset = create_small_subset(full_dataset)
    
    # 2. Split data
    print("\n[2/6] Splitting into train/val/test...")
    train_ds, val_ds = train_val_split(small_dataset)
    test_ds = small_dataset["test"]
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # 3. Tokenize
    print("\n[3/6] Tokenizing with BERT tokenizer...")
    preprocessor = BERTPreprocessor()
    train_tokenized, val_tokenized, test_tokenized = preprocessor.preprocess_datasets(
        train_ds, val_ds, test_ds
    )
    
    # 4. Train model
    print("\n[4/6] Fine-tuning BERT...")
    trainer = train_model(train_tokenized, val_tokenized, preprocessor.get_tokenizer())
    
    # 5. Evaluate
    print("\n[5/6] Evaluating on test set...")
    results = evaluate_model(trainer, test_tokenized)
    
    # 6. Test inference
    print("\n[6/6] Testing inference...")
    predictor = SentimentPredictor()
    
    test_reviews = [
        "This movie was fantastic!",
        "Worst film ever made.",
        "Pretty average, nothing special."
    ]
    
    for review in test_reviews:
        result = predictor.predict(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {result['label']} ({result['confidence']:.2%})")
    
    print("\n✅ Pipeline complete!")
    
    # Save metrics to file
    with open("training_results.txt", "w") as f:
        f.write("Training Results\n")
        f.write("="*50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return trainer, results

if __name__ == "__main__":
    main()