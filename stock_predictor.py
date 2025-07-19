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
        self.training_data = None
        self.training_labels = None
        
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
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
            hist['EMA_26'] = hist['Close'].ewm(span=26).mean()
            hist['RSI'] = self.calculate_rsi(hist['Close'])
            hist['MACD'] = self.calculate_macd(hist['Close'])
            hist['BB_upper'], hist['BB_lower'] = self.calculate_bollinger_bands(hist['Close'])
            hist['ATR'] = self.calculate_atr(hist)
            hist['OBV'] = self.calculate_obv(hist)
            
            # Calculate price changes
            hist['Price_Change'] = hist['Close'].pct_change()
            hist['Price_Change_5d'] = hist['Close'].pct_change(5)
            hist['Price_Change_20d'] = hist['Close'].pct_change(20)
            hist['Volume_Change'] = hist['Volume'].pct_change()
            hist['Volume_Change_5d'] = hist['Volume'].pct_change(5)
            
            # Calculate support and resistance levels
            support, resistance = self.calculate_support_resistance(hist)
            
            # Get current data
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # Calculate price distance from key levels
            price_to_sma_200_pct = ((current_price - hist['SMA_200'].iloc[-1]) / hist['SMA_200'].iloc[-1]) * 100 if not pd.isna(hist['SMA_200'].iloc[-1]) else 0
            price_to_support_pct = ((current_price - support) / support) * 100 if support > 0 else 0
            price_to_resistance_pct = ((resistance - current_price) / current_price) * 100 if resistance > 0 else 0
            
            # Prepare historical price data for charts
            history = []
            for i in range(min(30, len(hist))):
                idx = len(hist) - i - 1
                if idx >= 0:
                    history.append({
                        'date': hist.index[idx].strftime('%Y-%m-%d'),
                        'close': round(hist['Close'].iloc[idx], 2),
                        'volume': int(hist['Volume'].iloc[idx])
                    })
            history.reverse()
            
            return {
                'current_price': round(current_price, 2),
                'current_volume': current_volume,
                'price_change': round(hist['Price_Change'].iloc[-1] * 100, 2),
                'price_change_5d': round(hist['Price_Change_5d'].iloc[-1] * 100, 2),
                'price_change_20d': round(hist['Price_Change_20d'].iloc[-1] * 100, 2),
                'sma_20': round(hist['SMA_20'].iloc[-1], 2),
                'sma_50': round(hist['SMA_50'].iloc[-1], 2),
                'sma_200': round(hist['SMA_200'].iloc[-1], 2) if not pd.isna(hist['SMA_200'].iloc[-1]) else 0,
                'ema_12': round(hist['EMA_12'].iloc[-1], 2),
                'ema_26': round(hist['EMA_26'].iloc[-1], 2),
                'rsi': round(hist['RSI'].iloc[-1], 2),
                'macd': round(hist['MACD'].iloc[-1], 4),
                'bb_upper': round(hist['BB_upper'].iloc[-1], 2),
                'bb_lower': round(hist['BB_lower'].iloc[-1], 2),
                'atr': round(hist['ATR'].iloc[-1], 2),
                'obv': int(hist['OBV'].iloc[-1]),
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'price_to_sma_200_pct': round(price_to_sma_200_pct, 2),
                'price_to_support_pct': round(price_to_support_pct, 2),
                'price_to_resistance_pct': round(price_to_resistance_pct, 2),
                'historical_data': hist,
                'history': history
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
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    def calculate_obv(self, data):
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=data.index)
        obv.iloc[0] = 0
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
        
    def calculate_support_resistance(self, data, window=20):
        """Calculate support and resistance levels"""
        # Simple method: use recent lows for support and highs for resistance
        recent_data = data.iloc[-window:]
        
        # Support: average of 3 lowest lows
        support_levels = recent_data['Low'].nsmallest(3)
        support = support_levels.mean()
        
        # Resistance: average of 3 highest highs
        resistance_levels = recent_data['High'].nlargest(3)
        resistance = resistance_levels.mean()
        
        return support, resistance
    
    def prepare_features(self, stock_data, sentiment_score):
        """Prepare features for prediction"""
        hist = stock_data['historical_data']
        
        # Technical features
        features = {
            'price_change_1d': hist['Price_Change'].iloc[-1],
            'price_change_5d': hist['Price_Change_5d'].iloc[-1],
            'price_change_20d': hist['Price_Change_20d'].iloc[-1],
            'volume_change': hist['Volume_Change'].iloc[-1],
            'volume_change_5d': hist['Volume_Change_5d'].iloc[-1],
            'rsi': hist['RSI'].iloc[-1],
            'macd': hist['MACD'].iloc[-1],
            'sma_20_50_ratio': hist['SMA_20'].iloc[-1] / hist['SMA_50'].iloc[-1],
            'sma_50_200_ratio': hist['SMA_50'].iloc[-1] / hist['SMA_200'].iloc[-1] if not pd.isna(hist['SMA_200'].iloc[-1]) else 1,
            'bb_position': (hist['Close'].iloc[-1] - hist['BB_lower'].iloc[-1]) / 
                          (hist['BB_upper'].iloc[-1] - hist['BB_lower'].iloc[-1]),
            'atr_volatility': hist['ATR'].iloc[-1] / hist['Close'].iloc[-1],
            'obv_change': (hist['OBV'].iloc[-1] - hist['OBV'].iloc[-5]) / hist['OBV'].iloc[-5] if len(hist) > 5 and hist['OBV'].iloc[-5] != 0 else 0,
            'price_to_support': stock_data['price_to_support_pct'] / 100,
            'price_to_resistance': stock_data['price_to_resistance_pct'] / 100,
            'sentiment_score': sentiment_score
        }
        
        return features
    
    def train_model(self, training_data, training_labels):
        """
        Train the RandomForestClassifier model with provided data.
        """
        try:
            scaled_data = self.scaler.fit_transform(training_data)
            self.model.fit(scaled_data, training_labels)
            self.is_trained = True
            print("Model training completed.")
        except Exception as e:
            print(f"Error training model: {e}")
    
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
            
            # Use trained model if available
            if self.is_trained:
                feature_values = [features[key] for key in ['price_change_1d', 'price_change_5d', 'volume_change', 'rsi', 'macd', 'sma_ratio', 'bb_position', 'sentiment_score']]
                scaled_features = self.scaler.transform([feature_values])
                pred_proba = self.model.predict_proba(scaled_features)[0]
                pred_class = self.model.predict(scaled_features)[0]
                
                confidence = max(pred_proba) * 100
                if pred_class == 1:
                    prediction = 'RISE'
                    recommendation = 'BUY'
                elif pred_class == -1:
                    prediction = 'FALL'
                    recommendation = 'SELL'
                else:
                    prediction = 'STABLE'
                    recommendation = 'HOLD'
            else:
                # Enhanced rule-based prediction (fallback)
                prediction_score = 0
                
                # Technical analysis scoring
                # RSI (Oversold/Overbought)
                if features['rsi'] < 30:
                    prediction_score += 0.25  # Oversold - bullish
                elif features['rsi'] > 70:
                    prediction_score -= 0.25  # Overbought - bearish
                
                # MACD
                if features['macd'] > 0:
                    prediction_score += 0.15  # Positive MACD - bullish
                else:
                    prediction_score -= 0.15  # Negative MACD - bearish
                
                # Moving Average Trends
                if features['sma_20_50_ratio'] > 1.01:  # Short-term uptrend
                    prediction_score += 0.15
                elif features['sma_20_50_ratio'] < 0.99:  # Short-term downtrend
                    prediction_score -= 0.15
                    
                if features['sma_50_200_ratio'] > 1.01:  # Long-term uptrend
                    prediction_score += 0.1
                elif features['sma_50_200_ratio'] < 0.99:  # Long-term downtrend
                    prediction_score -= 0.1
                
                # Bollinger Bands
                if features['bb_position'] < 0.2:  # Near lower band - potential bounce
                    prediction_score += 0.1
                elif features['bb_position'] > 0.8:  # Near upper band - potential reversal
                    prediction_score -= 0.1
                    
                # Support/Resistance
                if features['price_to_support'] < 0.05:  # Close to support - bullish
                    prediction_score += 0.15
                if features['price_to_resistance'] < 0.05:  # Close to resistance - bearish
                    prediction_score -= 0.15
                    
                # Volume trends
                if features['volume_change_5d'] > 1.0:  # Volume increasing
                    if features['price_change_5d'] > 0:
                        prediction_score += 0.1  # Rising price with rising volume - bullish
                    else:
                        prediction_score -= 0.1  # Falling price with rising volume - bearish
                
                # Sentiment scoring with higher weight
                sentiment_score = features['sentiment_score']
                if sentiment_score > 0.2:
                    prediction_score += 0.25  # Strong positive sentiment
                elif sentiment_score > 0.1:
                    prediction_score += 0.15  # Moderate positive sentiment
                elif sentiment_score < -0.2:
                    prediction_score -= 0.25  # Strong negative sentiment
                elif sentiment_score < -0.1:
                    prediction_score -= 0.15  # Moderate negative sentiment
                
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
                'prediction_score': round(prediction_score, 3) if not self.is_trained else None,
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
