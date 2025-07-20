"""
Configuration settings for the Stock Notification Application.
"""
import os
from typing import List

# Flask Configuration
class Config:
    """Base configuration class."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'stocknotify-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Cache Configuration
    CACHE_TTL = 1800  # 30 minutes
    
    # Stock List - Indian stocks by default
    STOCK_LIST: List[str] = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "HDFC.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS"
    ]
    
    # API Keys (optional)
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    
    # Data Analysis Settings
    HISTORICAL_DAYS = 365  # Days of historical data to fetch
    PREDICTION_HORIZON = 5  # Days to predict into the future
    
    # Technical Analysis Settings
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    SMA_SHORT = 20
    SMA_LONG = 50
    
    # Sentiment Analysis Settings
    SENTIMENT_THRESHOLD = 0.1  # Threshold for positive/negative sentiment
    
    # Recommendation Settings
    BUY_THRESHOLD = 0.6
    SELL_THRESHOLD = 0.4
    CONFIDENCE_WEIGHTS = {
        'ml_models': 0.4,
        'technical': 0.3,
        'sentiment': 0.3
    }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    STOCK_LIST = ["AAPL", "GOOGL", "MSFT"]  # Use US stocks for testing

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 