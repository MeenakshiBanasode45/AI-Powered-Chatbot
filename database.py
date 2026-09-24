import sqlite3
import os
import datetime

DB_PATH = os.path.join('data', 'chatbot.db')

def init_db():
    """Initialize the database and create tables if they don't exist."""
    try:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                intent TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing database: {e}")

def log_conversation(user_message, bot_response, intent, confidence):
    """Log a conversation turn into the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_logs (user_message, bot_response, intent, confidence)
            VALUES (?, ?, ?, ?)
        ''', (user_message, bot_response, intent, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging conversation: {e}")

def get_chat_history():
    """Retrieve all chat logs ordered by timestamp descending."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id, user_message, bot_response, intent, confidence, timestamp FROM chat_logs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def clear_chat_history():
    """Delete all rows from chat_logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_logs')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return False
