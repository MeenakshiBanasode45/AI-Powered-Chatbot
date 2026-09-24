from flask import Flask, render_template, request, jsonify, session
import os
import secrets
from chatbot import Chatbot
import database

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Initialize Chatbot
chatbot_instance = Chatbot()

@app.route('/', methods=['GET'])
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
            
        message = data['message']
        if not message.strip():
            return jsonify({'error': 'Message cannot be empty'}), 400
            
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(8)
            
        session_id = session['session_id']
        result = chatbot_instance.get_response(message, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'An internal error occurred while processing your message.'}), 500

@app.route('/history', methods=['GET'])
def history():
    """Display chat history."""
    try:
        conversations = database.get_chat_history()
        return render_template('history.html', conversations=conversations)
    except Exception as e:
        app.logger.error(f"Error fetching history: {e}")
        return render_template('history.html', conversations=[])

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the chat history."""
    try:
        success = database.clear_chat_history()
        if success:
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Failed to clear history'}), 500
    except Exception as e:
        app.logger.error(f"Error clearing history: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
