from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import pandas as pd
from datetime import datetime
import traceback
import tempfile

from scraper import YouTubeScraper
from language_detector import LanguageDetector
from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor
from config import Config

app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)

# Initialize components
print("Initializing application components...")
try:
    scraper = YouTubeScraper()
    language_detector = LanguageDetector()
    sentiment_analyzer = SentimentAnalyzer()
    print("✓ All components initialized successfully")
except Exception as e:
    print(f"✗ Error initializing components: {e}")
    scraper = None
    language_detector = None
    sentiment_analyzer = None

# Store latest analysis for export
latest_analysis = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'components': {
            'scraper': scraper is not None,
            'language_detector': language_detector is not None,
            'sentiment_analyzer': sentiment_analyzer is not None
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Analyze YouTube video comments"""
    global latest_analysis
    
    try:
        # Get video URL from request
        data = request.get_json()
        video_url = data.get('url')
        
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"Analyzing video: {video_url}")
        print(f"{'='*60}")
        
        # Step 1: Scrape comments
        print("\n[1/3] Scraping comments...")
        scrape_result = scraper.scrape_comments(video_url)
        video_info = scrape_result['video_info']
        comments = scrape_result['comments']
        
        print(f"✓ Scraped {len(comments)} comments")
        
        if len(comments) == 0:
            return jsonify({'error': 'No comments found for this video'}), 400
        
        # Step 2: Detect languages
        print("\n[2/3] Detecting languages...")
        for i, comment in enumerate(comments):
            lang_code = language_detector.detect_language(comment['text'])
            lang_name = language_detector.get_language_name(lang_code)
            comments[i]['language'] = lang_code
            comments[i]['language_name'] = lang_name
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(comments)} comments")
        
        print(f"✓ Language detection complete")
        
        # Step 3: Analyze sentiment
        print("\n[3/3] Analyzing sentiment...")
        texts = [c['text'] for c in comments]
        sentiment_results = sentiment_analyzer.analyze_batch(texts)
        
        for i, result in enumerate(sentiment_results):
            comments[i]['sentiment'] = result['sentiment']
            comments[i]['confidence'] = result['confidence']
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(comments)} comments")
        
        print(f"✓ Sentiment analysis complete")
        
        # Step 4: Process data for frontend
        print("\n[4/4] Processing analytics...")
        
        sentiment_dist = DataProcessor.calculate_sentiment_distribution(comments)
        language_dist = DataProcessor.calculate_language_distribution(comments)
        weekly_traffic = DataProcessor.calculate_weekly_traffic(comments)
        avg_confidence = DataProcessor.calculate_average_confidence(comments)
        top_comments = DataProcessor.get_top_comments(comments, limit=20)
        
        # Prepare response
        analysis_result = {
            'video_info': video_info,
            'overview': {
                'total_comments': len(comments),
                'sentiment_distribution': sentiment_dist,
                'average_confidence': avg_confidence,
                'languages_detected': len([l for l in language_dist if l['language'] != 'unknown'])
            },
            'sentiment_distribution': sentiment_dist,
            'language_distribution': language_dist,
            'weekly_traffic': weekly_traffic,
            'top_comments': top_comments,
            'all_comments': comments  # Send ALL comments to frontend
        }
        
        # Store for export
        latest_analysis = DataProcessor.prepare_export_data(
            video_info, comments, analysis_result['overview']
        )
        
        print(f"\n{'='*60}")
        print("✓ Analysis complete!")
        print(f"{'='*60}\n")
        
        return jsonify(analysis_result)
    
    except ValueError as e:
        print(f"✗ Error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Export analysis results as CSV"""
    global latest_analysis
    
    if not latest_analysis:
        return jsonify({'error': 'No analysis data available. Please analyze a video first.'}), 400
    
    try:
        # Create DataFrame
        df = pd.DataFrame(latest_analysis['comments'])
        
        # Use platform-independent temp directory
        import tempfile
        temp_dir = tempfile.gettempdir()
        
        # Save to CSV
        filename = f"youtube_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(temp_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"✓ CSV saved to: {filepath}")
        
        return send_file(filepath, as_attachment=True, download_name=filename, mimetype='text/csv')
    except Exception as e:
        print(f"✗ CSV Export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/api/export/json', methods=['GET'])
def export_json():
    """Export analysis results as JSON"""
    global latest_analysis
    
    if not latest_analysis:
        return jsonify({'error': 'No analysis data available. Please analyze a video first.'}), 400
    
    try:
        # Use platform-independent temp directory
        import tempfile
        temp_dir = tempfile.gettempdir()
        
        filename = f"youtube_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(latest_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON saved to: {filepath}")
        
        return send_file(filepath, as_attachment=True, download_name=filename, mimetype='application/json')
    except Exception as e:
        print(f"✗ JSON Export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500
# ========== FRONTEND SERVING ROUTES ========== 
# These should be at the SAME indentation level as other @app.route() functions!

@app.route('/')
def serve_index():
    """Serve landing page"""
    return send_file('../frontend/index.html')

@app.route('/index.html')
def serve_index_explicit():
    """Serve landing page explicitly"""
    return send_file('../frontend/index.html')

@app.route('/dashboard.html')
def serve_dashboard():
    """Serve dashboard page"""
    return send_file('../frontend/dashboard.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    return send_file(f'../frontend/css/{filename}')

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files"""
    return send_file(f'../frontend/js/{filename}')

# ========== END FRONTEND ROUTES ==========

if __name__ == '__main__':
    if scraper is None or language_detector is None or sentiment_analyzer is None:
        print("\n⚠️  WARNING: Some components failed to initialize!")
        print("Please check your configuration and model files.\n")
    else:
        print("\n" + "="*60)
        print("🚀 YouTube Sentiment Analysis Server")
        print("="*60)
        print(f"Server running at: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
        print(f"Frontend should connect to this address")
        print("="*60 + "\n")
    
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )