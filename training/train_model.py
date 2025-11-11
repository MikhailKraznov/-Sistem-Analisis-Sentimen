import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

class SentimentTrainer:
    def __init__(self, labeled_csv_path):
        print("Initializing Sentiment Trainer...")
        
        # Load labeled dataset
        self.df = pd.read_csv(labeled_csv_path)
        print(f"Loaded {len(self.df)} labeled comments")
        
        # Sentiment to ID mapping
        self.sentiment2id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id2sentiment = {v: k for k, v in self.sentiment2id.items()}
        
        # Check CUDA compatibility
        if torch.cuda.is_available():
            try:
                # Test CUDA with simple operation
                test_tensor = torch.rand(5, 3).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ CUDA test passed")
            except RuntimeError as e:
                print("⚠️ CUDA error detected!")
                print(f"Error: {e}")
                print("\nSolutions:")
                print("1. Reinstall PyTorch to match your CUDA version")
                print("2. Update NVIDIA drivers")
                print("3. Or force CPU training (add os.environ['CUDA_VISIBLE_DEVICES']='-1' at top)")
                raise
        
        # Load tokenizer and model
        print("Loading XLM-RoBERTa base model...")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=3
        )
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # GPU info
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def prepare_data(self):
        """Prepare dataset for training"""
        print("\nPreparing dataset...")
        
        # Convert sentiments to IDs
        self.df['label'] = self.df['sentiment'].map(self.sentiment2id)
        
        # Split into train/validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.df['comment'].tolist(),
            self.df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=self.df['label']
        )
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'label': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'label': val_labels
        })
        
        # Tokenize - OPTIMIZED: Remove padding here, do dynamic padding later
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=128,  # ✅ Further reduced for 4GB GPU (was 256)
                # ✅ Remove padding='max_length' for dynamic padding
            )
        
        print("Tokenizing dataset...")
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        print("Dataset preparation complete!")
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, output_dir='../models/xlm_roberta_youtube'):
        """Train the model - OPTIMIZED FOR SPEED"""
        print("\nStarting training...")
        print("🚀 Optimizations for 4GB GPU:")
        print("  ✅ Mixed precision (fp16)")
        print("  ✅ Optimized batch size (8)")
        print("  ✅ Gradient accumulation (4x)")
        print("  ✅ Parallel data loading")
        print("  ✅ Dynamic padding")
        print("  ✅ Max length 128 tokens")
        print("  ⚠️  Gradient checkpointing disabled (CUDA compatibility)")
        
        # ✅ OPTIMIZED Training arguments for 4GB GPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            
            # ✅ OPTIMIZED FOR 4GB GPU: Smaller batch sizes
            per_device_train_batch_size=8,   # Safe for 4GB GPU
            per_device_eval_batch_size=16,   # Safe for 4GB GPU
            
            # ✅ SPEED OPTIMIZATION: Gradient accumulation for effective larger batch
            gradient_accumulation_steps=4,   # Effective batch = 8*4 = 32
            
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            
            # ✅ SPEED OPTIMIZATION: Mixed precision training (2x faster!)
            fp16=True,  # Works well with 4GB GPU
            
            # ✅ MEMORY OPTIMIZATION: Commented out due to CUDA compatibility
            # gradient_checkpointing=True,  # Can cause CUDA errors on some setups
            
            # ✅ SPEED OPTIMIZATION: Parallel data loading
            dataloader_num_workers=4,        # Load data in parallel
            dataloader_pin_memory=True,      # Faster GPU transfer
            
            # Logging and saving
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,                # Log more frequently for monitoring
            eval_strategy="steps",
            eval_steps=250,                  # Evaluate less frequently
            save_strategy="steps",
            save_steps=250,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            report_to="none",
            
            # ✅ SPEED OPTIMIZATION: Disable unnecessary features
            disable_tqdm=False,  # Keep progress bar
            group_by_length=True,  # Group similar lengths for efficiency
        )
        
        # ✅ OPTIMIZED: Dynamic padding (faster than max_length padding)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        print("\n" + "="*50)
        print("Starting optimized training...")
        print("="*50 + "\n")
        
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n⚠️ GPU Out of Memory!")
                print("Solutions:")
                print("  1. Reduce batch size: per_device_train_batch_size=16")
                print("  2. Or reduce max_length: max_length=128")
                print("  3. Or disable fp16 if causing issues")
                raise
            else:
                raise
        
        # Save model and tokenizer
        print(f"\n💾 Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Final evaluation
        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)
        results = trainer.evaluate()
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        
        print("\n✅ Training complete!")
        print(f"📁 Model saved to: {output_dir}")

def main():
    """Main training pipeline"""
    
    # Path to labeled dataset
    labeled_csv = '../dataset/labeled/combined_10k_labeled.csv'
    
    if not os.path.exists(labeled_csv):
        print(f"❌ Error: {labeled_csv} not found!")
        print("Please run auto_label.py first to generate labeled dataset.")
        return
    
    # Check CUDA
    print("="*50)
    print("SYSTEM CHECK")
    print("="*50)
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ CUDA not available - training will be VERY slow")
        print("Consider using Google Colab with GPU!")
    print("="*50 + "\n")
    
    # Initialize trainer
    trainer = SentimentTrainer(labeled_csv)
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()