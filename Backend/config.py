import os

class Config:
    # YouTube API
    YOUTUBE_API_KEY = "AIzaSyDNCmdmZI-XKoBbXI7s_Y-YloipDLW4j9s"  # Replace with your API key
    
    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
    
    # Scraping limits
    MAX_COMMENTS = 1000
    COMMENTS_PER_REQUEST = 100
    
    # Flask settings
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")  # UBAH INI biar bisa diakses dari luar
    FLASK_PORT = int(os.getenv("PORT", 7860))  # UBAH INI untuk HF Spaces
    FLASK_DEBUG = False
    
    # CORS settings
    CORS_ORIGINS = '*'