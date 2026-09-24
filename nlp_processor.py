import nltk
import re
import string
import json
import os
import random

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class NLPProcessor:
    def __init__(self):
        """Initialize the NLP processor."""
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
        self.intents = self.load_intents()

    def load_intents(self):
        """Load intents from JSON file."""
        try:
            if os.path.exists('intents.json'):
                with open('intents.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {'intents': []}
        except Exception as e:
            print(f"Error loading intents: {e}")
            return {'intents': []}

    def preprocess(self, text):
        """Preprocess text: lowercase, remove punctuation, tokenize, lemmatize, remove stopwords."""
        try:
            # Lowercase
            text = text.lower()
            # Remove punctuation keeping alphanumeric and spaces
            text = re.sub(r'[^a-z0-9\s]', '', text)
            # Tokenize
            tokens = word_tokenize(text)
            # Lemmatize and remove stopwords
            processed = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            return ' '.join(processed)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return text

    def keyword_match(self, processed_input):
        """Match processed input with intents using Jaccard similarity."""
        try:
            input_tokens = set(processed_input.split())
            if not input_tokens:
                return ('unknown', 0.0)

            best_tag = 'unknown'
            best_score = 0.0

            for intent in self.intents.get('intents', []):
                for pattern in intent.get('patterns', []):
                    processed_pattern = self.preprocess(pattern)
                    pattern_tokens = set(processed_pattern.split())
                    
                    if not pattern_tokens:
                        continue
                        
                    intersection = input_tokens.intersection(pattern_tokens)
                    union = input_tokens.union(pattern_tokens)
                    score = len(intersection) / len(union) if union else 0.0
                    
                    if score > best_score:
                        best_score = score
                        best_tag = intent.get('tag', 'unknown')

            return (best_tag, best_score)
        except Exception as e:
            print(f"Error in keyword matching: {e}")
            return ('unknown', 0.0)

    def get_response(self, tag):
        """Get a response for a specific tag."""
        try:
            for intent in self.intents.get('intents', []):
                if intent.get('tag') == tag:
                    responses = intent.get('responses', [])
                    if responses:
                        return random.choice(responses)
            return self.get_fallback_response()
        except Exception as e:
            print(f"Error getting response: {e}")
            return self.get_fallback_response()

    def get_fallback_response(self):
        """Return a default fallback response."""
        return "I'm not quite sure how to help with that. Try asking me about Orders, Refunds, Payments, Delivery, Working hours, or Customer support!"
