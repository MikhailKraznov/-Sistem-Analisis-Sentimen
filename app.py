import gradio as gr
import json
from scraper import YouTubeScraper
from language_detector import LanguageDetector
from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor

# Initialize components
print("🔧 Initializing components...")
scraper = YouTubeScraper()
language_detector = LanguageDetector()
sentiment_analyzer = SentimentAnalyzer()
print("✅ All components loaded!")

def analyze_youtube_video(video_url):
    """Main analysis function for Gradio"""
    try:
        if not video_url:
            return {"error": "Please provide a YouTube URL"}
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {video_url}")
        print(f"{'='*60}")
        
        # Step 1: Scrape comments
        print("\n[1/3] Scraping comments...")
        scrape_result = scraper.scrape_comments(video_url)
        video_info = scrape_result['video_info']
        comments = scrape_result['comments']
        print(f"✅ Scraped {len(comments)} comments")
        
        if len(comments) == 0:
            return {"error": "No comments found"}
        
        # Step 2: Detect languages
        print("\n[2/3] Detecting languages...")
        for i, comment in enumerate(comments):
            lang_code = language_detector.detect_language(comment['text'])
            lang_name = language_detector.get_language_name(lang_code)
            comments[i]['language'] = lang_code
            comments[i]['language_name'] = lang_name
        print(f"✅ Language detection complete")
        
        # Step 3: Analyze sentiment
        print("\n[3/3] Analyzing sentiment...")
        texts = [c['text'] for c in comments]
        sentiment_results = sentiment_analyzer.analyze_batch(texts)
        
        for i, result in enumerate(sentiment_results):
            comments[i]['sentiment'] = result['sentiment']
            comments[i]['confidence'] = result['confidence']
        print(f"✅ Sentiment analysis complete")
        
        # Step 4: Calculate analytics
        sentiment_dist = DataProcessor.calculate_sentiment_distribution(comments)
        language_dist = DataProcessor.calculate_language_distribution(comments)
        avg_confidence = DataProcessor.calculate_average_confidence(comments)
        
        # Prepare result
        result = {
            "video_title": video_info['title'],
            "channel": video_info['channel'],
            "total_comments": len(comments),
            "sentiment_distribution": {
                "positive": sentiment_dist['positive'],
                "neutral": sentiment_dist['neutral'],
                "negative": sentiment_dist['negative'],
                "positive_percent": sentiment_dist['positive_percent'],
                "neutral_percent": sentiment_dist['neutral_percent'],
                "negative_percent": sentiment_dist['negative_percent']
            },
            "average_confidence": f"{avg_confidence * 100:.1f}%",
            "languages_detected": len(language_dist),
            "top_languages": [
                f"{lang['language']}: {lang['count']} ({lang['percentage']}%)" 
                for lang in language_dist[:5]
            ],
            "sample_comments": [
                {
                    "text": c['text'][:100] + "..." if len(c['text']) > 100 else c['text'],
                    "sentiment": c['sentiment'],
                    "language": c['language_name'],
                    "confidence": f"{c['confidence']*100:.1f}%"
                }
                for c in comments[:10]
            ]
        }
        
        print(f"\n{'='*60}")
        print("✅ Analysis complete!")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}

# Create Gradio Interface
with gr.Blocks(title="YouTube Sentiment Analysis - UAT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 YouTube Sentiment Analysis")
    gr.Markdown("### Analyze sentiment and language distribution of YouTube comments")
    gr.Markdown("**For UAT Testing** - Enter a YouTube video URL below")
    
    with gr.Row():
        video_input = gr.Textbox(
            label="YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=4
        )
        analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)
    
    output = gr.JSON(label="Analysis Results")
    
    analyze_btn.click(
        fn=analyze_youtube_video,
        inputs=video_input,
        outputs=output
    )
    
    gr.Examples(
        examples=[
            ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        ],
        inputs=video_input
    )
    
    gr.Markdown("---")
    gr.Markdown("**Note:** Analysis may take 30-60 seconds for videos with many comments.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)