import json
import sqlite3
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
import time

# --- Database Interaction ---
def save_article_to_db(article_data):
    """Saves a single article to the database, avoiding duplicates based on URL."""
    conn = sqlite3.connect('newsletter_content.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO articles (source_name, title, url, summary, publication_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            article_data['source_name'],
            article_data['title'],
            article_data['url'],
            article_data['summary'],
            article_data.get('publication_date')
        ))
        conn.commit()
        print(f"SUCCESS: Saved article '{article_data['title']}'")
    except sqlite3.IntegrityError:
        # This error occurs if the URL (marked as UNIQUE) already exists.
        print(f"INFO: Article '{article_data['title']}' already exists. Skipping.")
    except Exception as e:
        print(f"ERROR: Could not save article '{article_data['title']}'. Reason: {e}")
    finally:
        conn.close()

# --- Scraping and Parsing ---
def fetch_articles_from_source(source):
    """Fetches articles from a single source, either via RSS or HTML scraping."""
    print(f"\n--- Processing source: {source['name']} ---")
    
    # Method 1: RSS Feed (Preferred)
    if source.get('feed'):
        feed = feedparser.parse(source['feed'])
        for entry in feed.entries:
            pub_date_parsed = time.strftime('%Y-%m-%d %H:%M:%S', entry.published_parsed) if hasattr(entry, 'published_parsed') else datetime.now().isoformat()
            
            article = {
                'source_name': source['name'],
                'title': entry.title,
                'url': entry.link,
                'summary': entry.summary if hasattr(entry, 'summary') else '',
                'publication_date': pub_date_parsed
            }
            save_article_to_db(article)
            
    # Method 2: HTML Scraping (Fallback for specific sites)
    # NOTE: This requires custom logic for each site.
    elif source['name'] == 'Anthropic News':
        try:
            response = requests.get(source['url'], headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # CUSTOM SCRAPING LOGIC FOR ANTHROPIC
            # This is an example and might need updates if the site structure changes.
            for item in soup.select('a.Card_cardLink__K0b1n'):
                title_element = item.select_one('h3')
                summary_element = item.select_one('p')
                
                if title_element and summary_element:
                    title = title_element.get_text(strip=True)
                    url = "https://www.anthropic.com" + item['href']
                    summary = summary_element.get_text(strip=True)
                    
                    article = {
                        'source_name': source['name'],
                        'title': title,
                        'url': url,
                        'summary': summary,
                        'publication_date': datetime.now().isoformat() # HTML pages often lack a standard date format
                    }
                    save_article_to_db(article)
        except Exception as e:
            print(f"ERROR: Failed to scrape {source['name']}. Reason: {e}")


def run_collection_pipeline():
    """Main function to run the entire data collection pipeline."""
    # Ensure database exists
    setup_database()

    # Load sources from the configuration file
    with open('sources.json', 'r') as f:
        sources_config = json.load(f)
    
    for source in sources_config['sources']:
        fetch_articles_from_source(source)
    
    print("\n--- Data collection cycle complete. ---")

# To run the collection process immediately:
# run_collection_pipeline()