# src/storage/source_tracker.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class SourceUsage(Base):
    __tablename__ = 'source_usage'

    id = Column(Integer, primary_key=True)
    source_url = Column(String, nullable=False)
    article_url = Column(String, nullable=False)
    title = Column(String, nullable=False)
    used_in_newsletter = Column(String)  # newsletter_id when used
    credibility_score = Column(Float)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    used_at = Column(DateTime)
