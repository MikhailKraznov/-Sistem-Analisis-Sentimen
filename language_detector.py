from langdetect import detect, DetectorFactory, LangDetectException
import re

# Set seed for consistent results
DetectorFactory.seed = 0

class LanguageDetector:
    def __init__(self):
        self.language_names = {
            'en': 'English',
            'es': 'Spanish',
            'id': 'Indonesian',
            'pt': 'Portuguese',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ar': 'Arabic',
            'ru': 'Russian',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'tr': 'Turkish',
            'pl': 'Polish',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish'
        }
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            # Clean text
            cleaned = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', str(text))
            if len(cleaned.strip()) < 3:
                return 'unknown'
            
            lang_code = detect(cleaned)
            return lang_code
        except LangDetectException:
            return 'unknown'
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'unknown'
    
    def get_language_name(self, lang_code):
        """Get full language name from code"""
        return self.language_names.get(lang_code, lang_code.upper())
    
    def detect_batch(self, texts):
        """Detect languages for multiple texts"""
        results = []
        for text in texts:
            lang = self.detect_language(text)
            results.append({
                'code': lang,
                'name': self.get_language_name(lang)
            })
        return results