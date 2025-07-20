"""
News and social sentiment analysis with multiple sources and caching
"""
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logger.warning("NewsAPI not available. Install with: pip install newsapi-python")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("Feedparser not available. Install with: pip install feedparser")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available. Install with: pip install textblob")

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logger.warning("BeautifulSoup not available. Install with: pip install beautifulsoup4")

class NewsSentimentAnalyzer:
    def __init__(self):
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        self.newsapi = NewsApiClient(api_key=self.newsapi_key) if self.newsapi_key and NEWSAPI_AVAILABLE else None
        self.rss_feeds = {
            'reuters': 'http://feeds.reuters.com/reuters/businessNews',
            'economic_times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms'
        }
        
    def get_news_sentiment(self, symbol: str, company_name: str) -> Tuple[List[Dict], float]:
        """
        Get news and sentiment from multiple sources with caching
        """
        news_items = []
        
        # Try NewsAPI first
        if self.newsapi and NEWSAPI_AVAILABLE:
            try:
                news_items.extend(self._get_newsapi_articles(symbol, company_name))
            except Exception as e:
                logger.error(f"NewsAPI error for {symbol}: {e}")
        
        # Add RSS feed news
        if FEEDPARSER_AVAILABLE:
            try:
                news_items.extend(self._get_rss_news(company_name))
            except Exception as e:
                logger.error(f"RSS feed error for {symbol}: {e}")
        
        # Calculate overall sentiment
        sentiment = self._calculate_sentiment(news_items)
        
        return news_items, sentiment
    
    def _get_newsapi_articles(self, symbol: str, company_name: str) -> List[Dict]:
        """
        Fetch news from NewsAPI
        """
        if not NEWSAPI_AVAILABLE or not self.newsapi:
            return []
            
        try:
            response = self.newsapi.get_everything(
                q=f'"{symbol}" OR "{company_name}"',
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            return response.get('articles', [])
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []
    
    def _get_rss_news(self, company_name: str) -> List[Dict]:
        """
        Fetch news from RSS feeds
        """
        if not FEEDPARSER_AVAILABLE:
            return []
            
        news_items = []
        for source, url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if company_name.lower() in entry.title.lower():
                        news_items.append({
                            'title': entry.title,
                            'description': entry.get('description', ''),
                            'url': entry.link,
                            'source': {'name': source}
                        })
            except Exception as e:
                logger.error(f"RSS feed error for {url}: {e}")
        
        return news_items
    
    def _calculate_sentiment(self, news_items: List[Dict]) -> float:
        """
        Calculate overall sentiment score from news items
        """
        if not news_items or not TEXTBLOB_AVAILABLE:
            return 0.0
            
        try:
            sentiments = []
            for item in news_items:
                text = f"{item['title']} {item.get('description', '')}"
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            return sum(sentiments) / len(sentiments) if sentiments else 0.0
        except Exception as e:
            logger.error(f"Sentiment calculation failed: {e}")
            return 0.0
    
    def _fallback_sentiment(self, text: str) -> float:
        """
        Simple fallback sentiment analysis using keyword matching
        """
        try:
            positive_words = ['positive', 'good', 'great', 'excellent', 'profit', 'gain', 'up', 'rise', 'growth']
            negative_words = ['negative', 'bad', 'poor', 'loss', 'down', 'fall', 'decline', 'risk', 'concern']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count == 0 and negative_count == 0:
                return 0.0
            elif positive_count > negative_count:
                return 0.3
            elif negative_count > positive_count:
                return -0.3
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Fallback sentiment failed: {e}")
            return 0.0
