import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
import os

class SentimentAnalyzer:
    def __init__(self, model_path=None, force_cpu=False):
        self.model_path = model_path or Config.MODEL_PATH

        if not os.path.exists(self.model_path):
            print(f"⚠️ Fine-tuned model not found at {self.model_path}")
            print("Using pretrained model instead...")
            self.model_path = "veisg/xlm-roberta-youtube-sentiment"

        print(f"Loading sentiment model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Smart device selection with CUDA error handling
        if force_cpu:
            self.device = torch.device("cpu")
            print("✓ Forced CPU mode")
        elif torch.cuda.is_available():
            try:
                # Test CUDA before using it
                torch.cuda.init()
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                self.device = torch.device("cuda")
                print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"⚠️ CUDA available but not working: {e}")
                print("✓ Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            print("✓ Using CPU (CUDA not available)")

        self.model.to(self.device)
        self.model.eval()

        self.id2sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}
        print(f"✓ Model loaded on {self.device}")

    def analyze_batch(self, texts, batch_size=16):
        """Analyze sentiment for multiple texts (optimized & safe for GPU)"""
        results = []

        # Filter out empty texts
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return []

        # Reduce batch size for CPU to avoid memory issues
        if self.device.type == "cpu":
            batch_size = min(batch_size, 8)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Tokenize entire batch at once
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_classes = torch.argmax(predictions, dim=1)

                # Move results to CPU for processing
                predictions_cpu = predictions.cpu()
                predicted_classes_cpu = predicted_classes.cpu()

                # Convert each to readable format
                for j, text in enumerate(batch):
                    label_id = predicted_classes_cpu[j].item()
                    sentiment = self.id2sentiment[label_id]
                    confidence = predictions_cpu[j][label_id].item()

                    results.append({
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'scores': {
                            'negative': predictions_cpu[j][0].item(),
                            'neutral': predictions_cpu[j][1].item(),
                            'positive': predictions_cpu[j][2].item()
                        }
                    })

                # Clear GPU cache if using CUDA
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA" in str(e) or "illegal" in str(e):
                    print(f"✗ CUDA error: {e}")
                    print("⚠️ Switching to CPU mode...")
                    
                    # Move model to CPU
                    self.model.to("cpu")
                    self.device = torch.device("cpu")
                    
                    # Retry the batch on CPU
                    try:
                        inputs = self.tokenizer(
                            batch,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                            padding=True
                        ).to(self.device)

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            predicted_classes = torch.argmax(predictions, dim=1)

                        for j, text in enumerate(batch):
                            label_id = predicted_classes[j].item()
                            sentiment = self.id2sentiment[label_id]
                            confidence = predictions[j][label_id].item()

                            results.append({
                                'sentiment': sentiment,
                                'confidence': confidence,
                                'scores': {
                                    'negative': predictions[j][0].item(),
                                    'neutral': predictions[j][1].item(),
                                    'positive': predictions[j][2].item()
                                }
                            })
                    except Exception as inner_e:
                        print(f"✗ Error even on CPU: {inner_e}")
                        results.extend([{
                            'sentiment': 'neutral',
                            'confidence': 0.0,
                            'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
                        }] * len(batch))
                else:
                    print(f"✗ Unexpected error: {e}")
                    results.extend([{
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
                    }] * len(batch))

            except Exception as e:
                print(f"✗ Unexpected error during batch processing: {e}")
                results.extend([{
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
                }] * len(batch))

        return results