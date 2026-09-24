import random
import logging
from nlp_processor import NLPProcessor
import database

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import numpy as np
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: SentenceTransformer, torch, or numpy not available.")

class Chatbot:
    def __init__(self):
        """Initialize Chatbot with NLP and DB."""
        self.nlp = NLPProcessor()
        database.init_db()
        self.context = {}
        
        # Confidence thresholds
        self.HIGH = 0.65
        self.MEDIUM = 0.40
        
        self.transformer = None
        self.encoded_intents = {}
        
        if TRANSFORMER_AVAILABLE:
            try:
                self.transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self._encode_intents()
            except Exception as e:
                print(f"Warning: Failed to load transformer model. Error: {e}")
                self.transformer = None
                
    def _encode_intents(self):
        """Pre-encode intent patterns if transformer is available."""
        if not self.transformer:
            return
            
        try:
            for intent in self.nlp.intents.get('intents', []):
                tag = intent.get('tag')
                patterns = intent.get('patterns', [])
                if patterns:
                    embeddings = self.transformer.encode(patterns)
                    self.encoded_intents[tag] = embeddings
        except Exception as e:
            print(f"Error encoding intents: {e}")
            
    def get_transformer_match(self, user_input):
        """Find best match using transformer embeddings."""
        if not self.transformer or not self.encoded_intents:
            return ('unknown', 0.0)
            
        try:
            import numpy as np
            from numpy.linalg import norm
            
            user_embedding = self.transformer.encode(user_input)
            
            best_tag = 'unknown'
            best_score = 0.0
            
            for tag, embeddings in self.encoded_intents.items():
                for emb in embeddings:
                    # Cosine similarity
                    similarity = np.dot(user_embedding, emb) / (norm(user_embedding) * norm(emb))
                    if similarity > best_score:
                        best_score = float(similarity)
                        best_tag = tag
                        
            return (best_tag, best_score)
        except Exception as e:
            print(f"Error in transformer matching: {e}")
            return ('unknown', 0.0)
            
    def _check_context(self, user_message, session_id):
        """Check if message is a follow-up based on context."""
        try:
            session_context = self.context.get(session_id, [])
            if not session_context:
                return None
                
            last_turn = session_context[-1]
            if self._needs_clarification(last_turn.get('intent')):
                # Simple concatenation for context combination
                return f"{last_turn.get('user_message')} {user_message}"
            return None
        except Exception as e:
            print(f"Error checking context: {e}")
            return None
            
    def _update_context(self, session_id, user_message, intent, response):
        """Update conversation history for session."""
        try:
            if session_id not in self.context:
                self.context[session_id] = []
                
            self.context[session_id].append({
                'user_message': user_message,
                'intent': intent,
                'response': response
            })
            
            # Keep max 5 turns
            if len(self.context[session_id]) > 5:
                self.context[session_id].pop(0)
        except Exception as e:
            print(f"Error updating context: {e}")
            
    def _needs_clarification(self, intent):
        """Determine if an intent typically needs follow-up."""
        clarification_intents = ['product_price', 'track_order', 'return_status']
        return intent in clarification_intents
        
    def get_response(self, user_message, session_id='default'):
        """Main method to get a response for user input."""
        try:
            if not user_message or not str(user_message).strip():
                return {
                    'response': "Please type a message.",
                    'intent': 'empty',
                    'confidence': 0.0
                }
                
            clean_message = str(user_message).strip()
            
            # Context checking
            context_msg = self._check_context(clean_message, session_id)
            process_msg = context_msg if context_msg else clean_message
            
            # Transformer match
            trans_tag, trans_score = self.get_transformer_match(process_msg)
            
            # NLP match
            processed_input = self.nlp.preprocess(process_msg)
            nlp_tag, nlp_score = self.nlp.keyword_match(processed_input)
            
            # Pick best match
            if trans_score >= nlp_score:
                best_tag = trans_tag
                best_score = trans_score
            else:
                best_tag = nlp_tag
                best_score = nlp_score
                
            # Apply thresholds
            if best_score >= self.HIGH:
                response = self.nlp.get_response(best_tag)
                final_intent = best_tag
            elif best_score >= self.MEDIUM:
                response = f"I think you're asking about {best_tag.replace('_', ' ')}. {self.nlp.get_response(best_tag)}"
                final_intent = best_tag
            else:
                response = self.nlp.get_fallback_response()
                final_intent = 'unknown'
                
            self._update_context(session_id, clean_message, final_intent, response)
            database.log_conversation(clean_message, response, final_intent, best_score)
            
            return {
                'response': response,
                'intent': final_intent,
                'confidence': float(best_score)
            }
            
        except Exception as e:
            print(f"Error getting response: {e}")
            return {
                'response': self.nlp.get_fallback_response(),
                'intent': 'error',
                'confidence': 0.0
            }
