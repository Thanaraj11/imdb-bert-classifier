"""Fine-tuning BERT with Hugging Face Trainer"""

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from config.model_config import CONFIG
from data.preprocess import BERTPreprocessor
from models.bert_model import initialize_model

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary with accuracy and F1 score
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary")
    }

def train_model(train_dataset, val_dataset, tokenizer):
    """
    Fine-tune BERT on sentiment classification.
    
    Args:
        train_dataset: Tokenized training dataset
        val_dataset: Tokenized validation dataset
        tokenizer: BERT tokenizer for data collation
        
    Returns:
        Trained Trainer object
    """
    model, device = initialize_model()
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG.learning_rate,
        per_device_train_batch_size=CONFIG.batch_size,
        per_device_eval_batch_size=CONFIG.eval_batch_size,
        num_train_epochs=CONFIG.num_epochs,
        weight_decay=CONFIG.weight_decay,
        fp16=CONFIG.fp16,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=CONFIG.seed,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("\n🚀 Starting fine-tuning...")
    trainer.train()
    
    # Save the model
    trainer.save_model(CONFIG.output_dir)
    print(f"\n✅ Model saved to {CONFIG.output_dir}")
    
    return trainer

def evaluate_model(trainer, test_dataset):
    """Evaluate trained model on test set"""
    print("\n📊 Evaluating on test set...")
    results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n=== Test Results ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    return results