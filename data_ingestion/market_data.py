"""
Market data fetcher using multiple sources with fallback support
"""
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import requests
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    logger.warning("Alpha Vantage not available. Install with: pip install alpha-vantage")

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logger.warning("FRED API not available. Install with: pip install fredapi")

class MarketDataFetcher:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
    @lru_cache(maxsize=100)
    def get_stock_data(self, symbol: str, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch stock data with fallback support
        """
        try:
            # Try yfinance first
            data = self._fetch_from_yfinance(symbol, start_date, end_date)
            if data is not None and not data.empty:
                return data
            
            # Fallback to Alpha Vantage if available
            if ALPHA_VANTAGE_AVAILABLE:
                return self._fetch_from_alpha_vantage(symbol)
            else:
                logger.warning(f"Alpha Vantage not available, using yfinance only for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_from_yfinance(self, symbol: str, start_date: Optional[datetime], 
                           end_date: Optional[datetime]) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if data.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return pd.DataFrame()
            return data
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_from_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage as backup
        """
        if not ALPHA_VANTAGE_AVAILABLE:
            logger.warning("Alpha Vantage not available")
            return pd.DataFrame()
            
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not found")
            return pd.DataFrame()
            
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            data, _ = ts.get_daily(symbol=symbol, outputsize='full')
            return data
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_index_data(self, index_symbol: str) -> pd.DataFrame:
        """
        Fetch index data from Stooq
        """
        try:
            url = f"https://stooq.com/q/d/l/?s={index_symbol}&i=d"
            data = pd.read_csv(url)
            return data
        except Exception as e:
            logger.error(f"Stooq fetch failed for {index_symbol}: {e}")
            return pd.DataFrame()

    def get_macro_data(self, indicator: str) -> pd.DataFrame:
        """
        Fetch macro data from FRED
        """
        if not FRED_AVAILABLE:
            logger.warning("FRED API not available")
            return pd.DataFrame()
            
        if not self.fred_api_key:
            logger.warning("FRED API key not found")
            return pd.DataFrame()
            
        try:
            fred = Fred(api_key=self.fred_api_key)
            data = fred.get_series(indicator)
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"FRED fetch failed for {indicator}: {e}")
            return pd.DataFrame()
