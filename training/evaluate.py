import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

class ModelEvaluator:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.id2sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.sentiment2id = {v: k for k, v in self.id2sentiment.items()}
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()
        
        return self.id2sentiment[predicted_class], confidence
    
    def evaluate_dataset(self, csv_path):
        """Evaluate model on labeled dataset"""
        print(f"\nEvaluating on {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
        predictions = []
        true_labels = []
        
        for idx, row in df.iterrows():
            pred_sentiment, confidence = self.predict(row['comment'])
            predictions.append(pred_sentiment)
            true_labels.append(row['sentiment'])
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples")
        
        # Convert to numeric for metrics
        pred_numeric = [self.sentiment2id[p] for p in predictions]
        true_numeric = [self.sentiment2id[t] for t in true_labels]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_numeric, pred_numeric, 
                                    target_names=['negative', 'neutral', 'positive']))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_numeric, pred_numeric)
        print(cm)
        
        # Accuracy
        accuracy = np.mean(np.array(pred_numeric) == np.array(true_numeric))
        print(f"\nOverall Accuracy: {accuracy:.2%}")

def main():
    model_path = '../models/xlm_roberta_youtube'
    test_csv = '../dataset/labeled/combined_10k_labeled.csv'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train_model.py first.")
        return
    
    evaluator = ModelEvaluator(model_path)
    evaluator.evaluate_dataset(test_csv)
    
    # Test on sample comments
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS:")
    print("="*60)
    
    test_comments = [
        "This is absolutely amazing! Best video ever!",
        "I hate this, waste of time",
        "It's okay, nothing special",
        "¡Increíble! Me encanta mucho",
        "Sangat bagus sekali!",
        "これは素晴らしいです"
    ]
    
    for comment in test_comments:
        sentiment, confidence = evaluator.predict(comment)
        print(f"\nComment: {comment}")
        print(f"Sentiment: {sentiment.upper()} (confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()