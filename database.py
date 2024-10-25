import sqlite3
from datetime import datetime

# Connect to SQLite database (or create it if it doesn't exist)
def init_db():
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    # Create a table for storing detection data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a new detection
def insert_detection(class_name, image_path):
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO detections (timestamp, class_name, image_path) VALUES (?, ?, ?)
    ''', (timestamp, class_name, image_path))
    conn.commit()
    conn.close()

init_db()  # Initialize the database
