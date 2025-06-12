# data_collection_pipeline.py

import sqlite3
import json
import requests
import feedparser
import pandas as pd
import schedule
import time
import logging
from bs4 import BeautifulSoup
from datetime import datetime

# --- 1. Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Database Schema Design & Storage ---
DB_NAME = 'newsletter_content.db'

def create_database_schema():
    """
    Creates the SQLite database and the 'articles' table if they don't exist.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            # Create table to store articles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    link TEXT NOT NULL UNIQUE,
                    summary TEXT,
                    source_name TEXT NOT NULL,
                    published_date DATETIME,
                    scraped_date DATETIME NOT NULL
                )
            ''')
            conn.commit()
            logging.info(f"Database '{DB_NAME}' and table 'articles' are ready.")
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise

def insert_articles_to_db(articles_df):
    """
    Inserts a DataFrame of articles into the SQLite database.
    Skips duplicates based on the unique 'link' constraint.
    """
    if articles_df.empty:
        logging.info("No new articles to insert.")
        return

    try:
        with sqlite3.connect(DB_NAME) as conn:
            # Use 'append' and 'if_exists' to add new data.
            # The UNIQUE constraint on the 'link' column will prevent duplicates.
            # We'll use a temporary table and INSERT OR IGNORE for robust duplicate handling.
            
            articles_df.to_sql('temp_articles', conn, if_exists='replace', index=False)
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO articles (title, link, summary, source_name, published_date, scraped_date)
                SELECT title, link, summary, source_name, published_date, scraped_date FROM temp_articles
            ''')
            
            inserted_count = cursor.rowcount
            conn.commit()
            logging.info(f"Successfully inserted {inserted_count} new articles into the database.")
            
    except sqlite3.Error as e:
        logging.error(f"Could not insert articles into database. Error: {e}")


# --- 3. Web Scraping & Content Extraction Pipeline ---

def fetch_from_rss(feed_url):
    """
    Fetches and parses articles from an RSS feed URL.
    """
    try:
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            logging.warning(f"Error parsing feed {feed_url}: {feed.bozo_exception}")
            return []
            
        articles = []
        for entry in feed.entries:
            # Safely get published_parsed, fallback to None
            published_tuple = entry.get('published_parsed')
            published_date = datetime.fromtimestamp(time.mktime(published_tuple)) if published_tuple else datetime.now()

            articles.append({
                'title': entry.get('title', 'No Title'),
                'link': entry.get('link', ''),
                'summary': BeautifulSoup(entry.get('summary', ''), 'html.parser').get_text(),
                'published_date': published_date
            })
        logging.info(f"Fetched {len(articles)} articles from RSS feed: {feed_url}")
        return articles
    except Exception as e:
        logging.error(f"Failed to fetch from RSS feed {feed_url}. Error: {e}")
        return []

def fetch_from_html(source):
    """
    Generic HTML scraper. 
    NOTE: This function is a placeholder and MUST be customized for each
    non-RSS site, as HTML structures vary widely.
    """
    url = source['url']
    logging.warning(f"HTML scraping for {url}. This requires custom logic.")
    # Example for Anthropic News (as of recent design)
    if 'anthropic.com' in url:
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code != 200:
                logging.error(f"Failed to fetch {url}, status code: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            # This selector is specific to Anthropic's news page and may break
            for post in soup.select('a.flex.w-full.flex-col'):
                title_tag = post.select_one('h3')
                link = post.get('href')
                
                if title_tag and link:
                    articles.append({
                        'title': title_tag.get_text(strip=True),
                        'link': f"https://www.anthropic.com{link}",
                        'summary': '', # Summary might not be available on the listing page
                        'published_date': datetime.now() # Date often not on listing page
                    })
            logging.info(f"Fetched {len(articles)} articles via HTML from: {url}")
            return articles
        except Exception as e:
            logging.error(f"Error scraping HTML from {url}. Error: {e}")
            return []
    return []

def run_collection_pipeline():
    """
    Main pipeline function to read sources, fetch content, and store it.
    """
    logging.info("--- Starting Data Collection Pipeline ---")
    
    # 1. Load sources from config file
    try:
        with open('sources.json', 'r') as f:
            sources_config = json.load(f)['sources']
    except FileNotFoundError:
        logging.error("'sources.json' not found. Please create it.")
        return
        
    all_articles = []
    
    # 2. Fetch content for each source
    for source in sources_config:
        source_name = source['name']
        logging.info(f"Processing source: {source_name}")
        
        articles = []
        if source.get('feed'):
            articles = fetch_from_rss(source['feed'])
        else:
            # Fallback to HTML scraping if no RSS feed
            articles = fetch_from_html(source)
            
        if articles:
            for article in articles:
                article['source_name'] = source_name
                article['scraped_date'] = datetime.now()
            all_articles.extend(articles)

    # 3. Convert to DataFrame and insert into DB
    if all_articles:
        articles_df = pd.DataFrame(all_articles)
        # Ensure data types are correct before insertion
        articles_df['published_date'] = pd.to_datetime(articles_df['published_date'], errors='coerce')
        articles_df['scraped_date'] = pd.to_datetime(articles_df['scraped_date'])
        insert_articles_to_db(articles_df)

    logging.info("--- Data Collection Pipeline Finished ---")


# --- 4. Scheduling System ---

def main():
    # Create the database and table on first run
    create_database_schema()
    
    # Run the collection pipeline immediately on start
    run_collection_pipeline()

    # Schedule the pipeline to run once every day
    logging.info("Scheduling job to run daily.")
    schedule.every().day.at("08:00").do(run_collection_pipeline)

    # For testing, you can schedule it to run more frequently:
    # schedule.every(1).minutes.do(run_collection_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute if a scheduled job is due

if __name__ == '__main__':
    main()