import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AutoLabeler:
    def __init__(self):
        print("Loading pretrained XLM-RoBERTa model for auto-labeling...")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        
        # FORCE CPU USAGE - Safer and more stable
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully")
        
        # Sentiment mapping
        self.sentiment_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=1).item()
                confidence = predictions[0][predicted_class].item()
            
            return self.sentiment_map[predicted_class], confidence
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return 'neutral', 0.0
    
    def label_dataset(self, input_csv, output_csv):
        """Label entire dataset"""
        print(f"\nProcessing: {input_csv}")
        
        # Load data
        df = pd.read_csv(input_csv)
        
        # Check required columns
        if 'comment' not in df.columns:
            print(f"Error: 'comment' column not found in {input_csv}")
            return
        
        # Add sentiment columns
        sentiments = []
        confidences = []
        
        # Process with progress bar
        for comment in tqdm(df['comment'], desc="Auto-labeling"):
            sentiment, confidence = self.predict_sentiment(str(comment))
            sentiments.append(sentiment)
            confidences.append(confidence)
        
        df['sentiment'] = sentiments
        df['sentiment_confidence'] = confidences
        
        # Save
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Saved labeled dataset to: {output_csv}")
        
        # Show statistics
        print("\nSentiment Distribution:")
        print(df['sentiment'].value_counts())
        print(f"\nAverage Confidence: {df['sentiment_confidence'].mean():.2%}")
        
        return df

def main():
    """Auto-label all 5 category datasets"""
    
    labeler = AutoLabeler()
    
    # Define input/output paths
    categories = ['movie', 'gaming', 'music', 'tech', 'news']
    base_input_dir = '../dataset/raw'
    base_output_dir = '../dataset/labeled'
    
    all_dfs = []
    
    for category in categories:
        input_path = os.path.join(base_input_dir, f'{category}_comments.csv')
        output_path = os.path.join(base_output_dir, f'{category}_labeled.csv')
        
        if os.path.exists(input_path):
            df = labeler.label_dataset(input_path, output_path)
            if df is not None:
                df['category'] = category
                all_dfs.append(df)
        else:
            print(f"Warning: {input_path} not found, skipping...")
    
    # Combine all datasets
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_output = os.path.join(base_output_dir, 'combined_10k_labeled.csv')
        combined_df.to_csv(combined_output, index=False, encoding='utf-8')
        print(f"\n{'='*60}")
        print(f"Combined dataset saved to: {combined_output}")
        print(f"Total comments: {len(combined_df)}")
        print("\nOverall Sentiment Distribution:")
        print(combined_df['sentiment'].value_counts())
        print(f"\nOverall Average Confidence: {combined_df['sentiment_confidence'].mean():.2%}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()