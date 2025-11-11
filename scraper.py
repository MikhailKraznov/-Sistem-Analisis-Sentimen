from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import time
from config import Config

class YouTubeScraper:
    def __init__(self, api_key=None):
        self.api_key = api_key or Config.YOUTUBE_API_KEY
        if not self.api_key or self.api_key == "YOUR_YOUTUBE_API_KEY_HERE":
            raise ValueError("Please set a valid YouTube API key in config.py")
        
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.max_comments = Config.MAX_COMMENTS
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_video_info(self, video_id):
        """Get video metadata"""
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                return None
            
            video = response['items'][0]
            return {
                'title': video['snippet']['title'],
                'channel': video['snippet']['channelTitle'],
                'published_at': video['snippet']['publishedAt'],
                'view_count': int(video['statistics'].get('viewCount', 0)),
                'like_count': int(video['statistics'].get('likeCount', 0)),
                'comment_count': int(video['statistics'].get('commentCount', 0))
            }
        except HttpError as e:
            print(f"Error fetching video info: {e}")
            return None
    
    def scrape_comments(self, video_url, progress_callback=None):
        """Scrape comments from YouTube video"""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
    
    # Get video info
        video_info = self.get_video_info(video_id)
        if not video_info:
            raise ValueError("Video not found or unavailable")
    
        comments = []
        nextPageToken = None
        total_fetched = 0
    
        print(f"📊 Target: {self.max_comments} comments")  # Debug line
    
        try:
            while total_fetched < self.max_comments:
            # Calculate how many to fetch this iteration
                remaining = self.max_comments - total_fetched
                fetch_size = min(Config.COMMENTS_PER_REQUEST, remaining)
            
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=fetch_size,  # Use calculated size
                    pageToken=nextPageToken,
                    textFormat='plainText',
                    order='relevance'
                )
            
                response = request.execute()
            
                for item in response['items']:
                    comment_data = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'text': comment_data['textDisplay'],
                        'author': comment_data['authorDisplayName'],
                        'likes': comment_data.get('likeCount', 0),
                        'published_at': comment_data['publishedAt']
                    })
            
                total_fetched = len(comments)
                print(f"  Fetched: {total_fetched}/{self.max_comments}")  # Progress
            
            # Progress callback
                if progress_callback:
                    progress_callback(total_fetched, self.max_comments)
            
            # Check for next page
                nextPageToken = response.get('nextPageToken')
                if not nextPageToken:
                    print(f"  No more comments available (got {total_fetched})")
                    break
            
                time.sleep(0.5)  # Rate limiting
    
        except HttpError as e:
            if e.resp.status == 403:
             raise ValueError("Comments are disabled for this video")
            else:
                raise ValueError(f"Error scraping comments: {str(e)}")
    
        print(f"✓ Total scraped: {len(comments)} comments")
    
        return {
            'video_info': video_info,
            'comments': comments,
            'total_comments': len(comments)
        }