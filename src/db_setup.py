import sqlite3

def setup_database():
    """Creates the SQLite database and the articles table if they don't exist."""
    conn = sqlite3.connect('newsletter_content.db')
    cursor = conn.cursor()
    
    # Create table for storing articles
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            summary TEXT,
            raw_content TEXT,
            publication_date TEXT,
            fetched_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_processed BOOLEAN DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database setup complete. 'newsletter_content.db' is ready.")

# Run this once to create your database file
# setup_database()