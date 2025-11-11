from datetime import datetime, timedelta
from collections import Counter
import pandas as pd

class DataProcessor:
    @staticmethod
    def calculate_sentiment_distribution(comments):
        """Calculate sentiment distribution"""
        sentiments = [c['sentiment'] for c in comments]
        total = len(sentiments)
        
        if total == 0:
            return {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'positive_percent': 0,
                'neutral_percent': 0,
                'negative_percent': 0
            }
        
        counter = Counter(sentiments)
        
        return {
            'positive': counter.get('positive', 0),
            'neutral': counter.get('neutral', 0),
            'negative': counter.get('negative', 0),
            'positive_percent': round(counter.get('positive', 0) / total * 100, 1),
            'neutral_percent': round(counter.get('neutral', 0) / total * 100, 1),
            'negative_percent': round(counter.get('negative', 0) / total * 100, 1)
        }
    
    @staticmethod
    def calculate_language_distribution(comments):
        """Calculate language distribution"""
        languages = [c['language'] for c in comments]
        counter = Counter(languages)
        total = len(languages)
        
        distribution = []
        for lang, count in counter.most_common():
            if lang != 'unknown':
                distribution.append({
                    'language': lang,
                    'count': count,
                    'percentage': round(count / total * 100, 1)
                })
        
        return distribution
    
    @staticmethod
    def calculate_weekly_traffic(comments):
        """Calculate weekly comment traffic - FIRST 4 WEEKS ONLY after upload"""
        from datetime import timedelta
    
    # Parse timestamps
        dates = []
        for comment in comments:
            try:
                dt = datetime.fromisoformat(comment['published_at'].replace('Z', '+00:00'))
                dates.append(dt)
            except:
                continue
    
        if not dates:
            return []
    
    # Sort dates to find upload date
        dates.sort()
        upload_date = dates[0]  # First comment = approximate upload date
    
    # Define 4 weeks from upload
        week_1_end = upload_date + timedelta(weeks=1)
        week_2_end = upload_date + timedelta(weeks=2)
        week_3_end = upload_date + timedelta(weeks=3)
        week_4_end = upload_date + timedelta(weeks=4)
    
    # Count comments in each week
        week_1 = sum(1 for d in dates if upload_date <= d < week_1_end)
        week_2 = sum(1 for d in dates if week_1_end <= d < week_2_end)
        week_3 = sum(1 for d in dates if week_2_end <= d < week_3_end)
        week_4 = sum(1 for d in dates if week_3_end <= d < week_4_end)
    
    # Create traffic data
        traffic = [
            {
                'period': 'Week 1',
                'count': week_1,
                'date_range': f"{upload_date.strftime('%b %d')} - {week_1_end.strftime('%b %d')}"
            },
            {
                'period': 'Week 2',
                'count': week_2,
                'date_range': f"{week_1_end.strftime('%b %d')} - {week_2_end.strftime('%b %d')}"
            },
            {
                'period': 'Week 3',
                'count': week_3,
                'date_range': f"{week_2_end.strftime('%b %d')} - {week_3_end.strftime('%b %d')}"
            },
            {
                'period': 'Week 4',
                'count': week_4,
                'date_range': f"{week_3_end.strftime('%b %d')} - {week_4_end.strftime('%b %d')}"
            }
        ]
    
        return traffic

    @staticmethod
    def _calculate_hourly_traffic(dates):
        """Calculate hourly traffic (for same-day comments)"""
        from collections import Counter
        hour_counts = Counter([d.hour for d in dates])
    
        traffic = []
        for hour in sorted(hour_counts.keys()):
            traffic.append({
                'period': f'{hour:02d}:00',
                'count': hour_counts[hour]
            })
    
        return traffic[:12]  # Max 12 hours

    @staticmethod
    def _calculate_daily_traffic(dates, min_date, max_date):
        """Calculate daily traffic"""
        from collections import Counter
        date_counts = Counter([d.date() for d in dates])
    
        traffic = []
        current_date = min_date.date()
        end_date = max_date.date()
    
        while current_date <= end_date:
            count = date_counts.get(current_date, 0)
            traffic.append({
                'period': current_date.strftime('%b %d'),
                'count': count
        })
            current_date += timedelta(days=1)
    
        return traffic

    @staticmethod
    def _calculate_weekly_buckets(dates, min_date, max_date, num_weeks=4):
        """Calculate weekly buckets"""
        total_days = (max_date - min_date).days
        days_per_week = max(total_days // num_weeks, 1)
    
        weeks = [0] * num_weeks
    
        for date in dates:
            days_from_start = (date - min_date).days
            week_index = min(days_from_start // days_per_week, num_weeks - 1)
            weeks[week_index] += 1
    
        traffic = []
        for i in range(num_weeks):
        # Only include weeks with comments
            if weeks[i] > 0 or i == 0:  # Always show week 1
                start_day = i * days_per_week
                end_day = start_day + days_per_week
                start_date = (min_date + timedelta(days=start_day)).strftime('%b %d')
                end_date = (min_date + timedelta(days=min(end_day, total_days))).strftime('%b %d')
            
                traffic.append({
                    'period': f'{start_date} - {end_date}',
                    'count': weeks[i]
                })
    
        return traffic

    @staticmethod
    def _calculate_monthly_traffic(dates):
        """Calculate monthly traffic"""
        from collections import Counter
        month_year = Counter([(d.year, d.month) for d in dates])
    
        traffic = []
        for (year, month), count in sorted(month_year.items()):
            date_obj = datetime(year, month, 1)
            traffic.append({
                'period': date_obj.strftime('%B %Y'),
                'count': count
            })
    
        return traffic

    @staticmethod
    def _calculate_period_buckets(dates, min_date, max_date, num_periods=4):
        """Calculate equal time period buckets"""
        total_days = (max_date - min_date).days
        days_per_period = max(total_days // num_periods, 1)
    
        periods = [0] * num_periods
    
        for date in dates:
            days_from_start = (date - min_date).days
            period_index = min(days_from_start // days_per_period, num_periods - 1)
            periods[period_index] += 1
    
        traffic = []
        for i in range(num_periods):
        # Only include periods with comments
            if periods[i] > 0:
                traffic.append({
                    'period': f'Period {i + 1}',
                    'count': periods[i]
                })
    
        return traffic
    
    @staticmethod
    def calculate_average_confidence(comments):
        """Calculate average sentiment confidence"""
        confidences = [c.get('confidence', 0) for c in comments if c.get('confidence')]
        if not confidences:
            return 0.0
        return round(sum(confidences) / len(confidences), 2)
    
    @staticmethod
    def get_top_comments(comments, limit=100):
        """Get diverse top comments (by sentiment and language)"""
    # Separate by sentiment
        positive = [c for c in comments if c.get('sentiment') == 'positive']
        negative = [c for c in comments if c.get('sentiment') == 'negative']
        neutral = [c for c in comments if c.get('sentiment') == 'neutral']
    
    # Sort each by likes
        positive.sort(key=lambda x: x.get('likes', 0), reverse=True)
        negative.sort(key=lambda x: x.get('likes', 0), reverse=True)
        neutral.sort(key=lambda x: x.get('likes', 0), reverse=True)
    
    # Get diverse selection
    # 50% positive, 30% negative, 20% neutral (proportional distribution)
        num_positive = int(limit * 0.5)
        num_negative = int(limit * 0.3)
        num_neutral = limit - num_positive - num_negative
    
        top_comments = []
        top_comments.extend(positive[:num_positive])
        top_comments.extend(negative[:num_negative])
        top_comments.extend(neutral[:num_neutral])
    
    # If we don't have enough in a category, fill from others
        while len(top_comments) < limit:
            if len(positive) > num_positive:
                top_comments.append(positive[num_positive])
                num_positive += 1
            elif len(neutral) > num_neutral:
                top_comments.append(neutral[num_neutral])
                num_neutral += 1
            elif len(negative) > num_negative:
                top_comments.append(negative[num_negative])
                num_negative += 1
            else:
                break
    
    # Shuffle to mix languages and sentiments
        import random
        random.shuffle(top_comments)
    
        return top_comments[:limit]
    
    @staticmethod
    def prepare_export_data(video_info, comments, analysis_summary):
        """Prepare data for CSV/JSON export"""
        export_data = {
            'video_info': video_info,
            'analysis_summary': analysis_summary,
            'comments': []
        }
        
        for comment in comments:
            export_data['comments'].append({
                'text': comment.get('text', ''),
                'author': comment.get('author', ''),
                'likes': comment.get('likes', 0),
                'published_at': comment.get('published_at', ''),
                'language': comment.get('language', 'unknown'),
                'language_name': comment.get('language_name', 'Unknown'),
                'sentiment': comment.get('sentiment', 'neutral'),
                'confidence': comment.get('confidence', 0.0)
            })
        
        return export_data