# Stock prediction module using technical analysis and sentiment
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        """Initialize the stock predictor with ML model"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def get_stock_data(self, symbol, period='1y'):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period for data
            
        Returns:
            dict: Stock data including price, volume, and technical indicators
        """
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
            
            # Calculate technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = self.calculate_rsi(hist['Close'])
            hist['MACD'] = self.calculate_macd(hist['Close'])
            hist['BB_upper'], hist['BB_lower'] = self.calculate_bollinger_bands(hist['Close'])
            
            # Calculate price changes
            hist['Price_Change'] = hist['Close'].pct_change()
            hist['Volume_Change'] = hist['Volume'].pct_change()
            
            # Get current data
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            return {
                'current_price': round(current_price, 2),
                'current_volume': current_volume,
                'price_change': round(hist['Price_Change'].iloc[-1] * 100, 2),
                'sma_20': round(hist['SMA_20'].iloc[-1], 2),
                'sma_50': round(hist['SMA_50'].iloc[-1], 2),
                'rsi': round(hist['RSI'].iloc[-1], 2),
                'macd': round(hist['MACD'].iloc[-1], 4),
                'bb_upper': round(hist['BB_upper'].iloc[-1], 2),
                'bb_lower': round(hist['BB_lower'].iloc[-1], 2),
                'historical_data': hist
            }
            
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def prepare_features(self, stock_data, sentiment_score):
        """Prepare features for prediction"""
        hist = stock_data['historical_data']
        
        # Technical features
        features = {
            'price_change_1d': hist['Price_Change'].iloc[-1],
            'price_change_5d': hist['Close'].pct_change(5).iloc[-1],
            'volume_change': hist['Volume_Change'].iloc[-1],
            'rsi': hist['RSI'].iloc[-1],
            'macd': hist['MACD'].iloc[-1],
            'sma_ratio': hist['SMA_20'].iloc[-1] / hist['SMA_50'].iloc[-1],
            'bb_position': (hist['Close'].iloc[-1] - hist['BB_lower'].iloc[-1]) / 
                          (hist['BB_upper'].iloc[-1] - hist['BB_lower'].iloc[-1]),
            'sentiment_score': sentiment_score
        }
        
        return features
    
    def predict_stock_movement(self, symbol, sentiment_data):
        """
        Predict stock movement based on technical analysis and sentiment
        
        Args:
            symbol (str): Stock symbol
            sentiment_data (dict): Sentiment analysis results
            
        Returns:
            dict: Prediction results
        """
        try:
            # Get stock data
            stock_data = self.get_stock_data(symbol)
            if stock_data is None:
                return {'prediction': 'Unknown', 'confidence': 0, 'recommendation': 'Unable to analyze'}
            
            # Prepare features
            features = self.prepare_features(stock_data, sentiment_data['overall_sentiment'])
            
            # Simple rule-based prediction (can be enhanced with ML)
            prediction_score = 0
            
            # Technical analysis scoring
            if features['rsi'] < 30:
                prediction_score += 0.3  # Oversold
            elif features['rsi'] > 70:
                prediction_score -= 0.3  # Overbought
            
            if features['macd'] > 0:
                prediction_score += 0.2  # Positive MACD
            else:
                prediction_score -= 0.2  # Negative MACD
            
            if features['sma_ratio'] > 1:
                prediction_score += 0.2  # Price above SMA
            else:
                prediction_score -= 0.2  # Price below SMA
            
            # Sentiment scoring
            sentiment_score = features['sentiment_score']
            if sentiment_score > 0.1:
                prediction_score += 0.3
            elif sentiment_score < -0.1:
                prediction_score -= 0.3
            
            # Determine prediction
            if prediction_score > 0.3:
                prediction = 'RISE'
                confidence = min(abs(prediction_score) * 100, 95)
                recommendation = 'BUY'
            elif prediction_score < -0.3:
                prediction = 'FALL'
                confidence = min(abs(prediction_score) * 100, 95)
                recommendation = 'SELL'
            else:
                prediction = 'STABLE'
                confidence = 50
                recommendation = 'HOLD'
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 1),
                'recommendation': recommendation,
                'prediction_score': round(prediction_score, 3),
                'technical_indicators': {
                    'rsi': stock_data['rsi'],
                    'macd': stock_data['macd'],
                    'sma_20': stock_data['sma_20'],
                    'sma_50': stock_data['sma_50']
                }
            }
            
        except Exception as e:
            print(f"Error predicting stock movement: {e}")
            return {'prediction': 'Error', 'confidence': 0, 'recommendation': 'Unable to analyze'} 