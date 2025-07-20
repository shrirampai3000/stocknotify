"""
Data ingestion module for stock market data and news sentiment analysis.
"""

from .market_data import MarketDataFetcher
from .news_sentiment import NewsSentimentAnalyzer

__all__ = ['MarketDataFetcher', 'NewsSentimentAnalyzer']
