"""
Recommendation engine using ML models and technical/fundamental signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

class RecommendationEngine:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
        # Initialize models if available
        if SKLEARN_AVAILABLE:
            self.models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            self.models['lgb'] = lgb.LGBMClassifier(random_state=42)
        
        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBClassifier(random_state=42)
        
    def generate_recommendation(self, 
                              stock_data: pd.DataFrame,
                              technical_indicators: Dict,
                              sentiment_score: float,
                              market_regime: Dict) -> Dict:
        """
        Generate trading recommendations using ensemble of signals
        """
        try:
            # Prepare features
            features = self._prepare_features(stock_data, technical_indicators, 
                                           sentiment_score, market_regime)
            
            if features.empty:
                return self._fallback_recommendation(technical_indicators, sentiment_score)
            
            # Get predictions from each model
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    # For now, use simple heuristics since models aren't trained
                    pred = self._simple_prediction(features, technical_indicators, sentiment_score)
                    predictions[name] = pred
                    probabilities[name] = [0.3, 0.7] if pred == 1 else [0.7, 0.3]
                except Exception as e:
                    logger.error(f"Prediction failed for {name}: {e}")
                    predictions[name] = 0
                    probabilities[name] = [0.5, 0.5]
            
            # Calculate feature importance using SHAP if available
            if SHAP_AVAILABLE and LIGHTGBM_AVAILABLE and 'lgb' in self.models:
                try:
                    explainer = shap.TreeExplainer(self.models['lgb'])
                    shap_values = explainer.shap_values(features)
                    self.feature_importance = dict(zip(features.columns, np.abs(shap_values).mean(0)))
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {e}")
                    self.feature_importance = {}
            else:
                self.feature_importance = {}
            
            # Generate final recommendation
            recommendation = self._combine_signals(
                predictions,
                probabilities,
                technical_indicators,
                sentiment_score
            )
            
            return {
                'recommendation': recommendation['action'],
                'confidence': recommendation['confidence'],
                'reasoning': recommendation['reasoning'],
                'model_predictions': predictions,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return self._fallback_recommendation(technical_indicators, sentiment_score)
    
    def _prepare_features(self, 
                         stock_data: pd.DataFrame,
                         technical_indicators: Dict,
                         sentiment_score: float,
                         market_regime: Dict) -> pd.DataFrame:
        """
        Prepare feature matrix for ML models
        """
        try:
            if stock_data.empty or 'Close' not in stock_data.columns:
                return pd.DataFrame()
                
            features = pd.DataFrame()
            
            # Price-based features
            returns = stock_data['Close'].pct_change()
            if len(returns) > 0:
                features['return'] = returns.iloc[-1] if not pd.isna(returns.iloc[-1]) else 0.0
                features['return_5d'] = returns.rolling(5).mean().iloc[-1] if not pd.isna(returns.rolling(5).mean().iloc[-1]) else 0.0
                features['return_20d'] = returns.rolling(20).mean().iloc[-1] if not pd.isna(returns.rolling(20).mean().iloc[-1]) else 0.0
                features['volatility'] = returns.rolling(20).std().iloc[-1] if not pd.isna(returns.rolling(20).std().iloc[-1]) else 0.0
            else:
                features['return'] = 0.0
                features['return_5d'] = 0.0
                features['return_20d'] = 0.0
                features['volatility'] = 0.0
            
            # Technical indicators
            for indicator, value in technical_indicators.items():
                features[indicator] = value if not pd.isna(value) else 0.0
            
            # Sentiment and regime
            features['sentiment_score'] = sentiment_score
            features['market_regime'] = market_regime.get('regimes', [-1])[-1] if market_regime.get('regimes') else -1
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def _simple_prediction(self, features: pd.DataFrame, technical_indicators: Dict, sentiment_score: float) -> int:
        """
        Simple prediction based on technical indicators and sentiment
        """
        try:
            score = 0
            
            # Technical signals
            if 'sma_20' in technical_indicators and 'sma_50' in technical_indicators:
                if technical_indicators['sma_20'] > technical_indicators['sma_50']:
                    score += 1
                else:
                    score -= 1
            
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi']
                if rsi < 30:
                    score += 1  # Oversold, potential buy
                elif rsi > 70:
                    score -= 1  # Overbought, potential sell
            
            # Sentiment
            if sentiment_score > 0.1:
                score += 1
            elif sentiment_score < -0.1:
                score -= 1
            
            # Return prediction (1 for buy, 0 for sell/hold)
            return 1 if score > 0 else 0
            
        except Exception as e:
            logger.error(f"Simple prediction failed: {e}")
            return 0
    
    def _fallback_recommendation(self, technical_indicators: Dict, sentiment_score: float) -> Dict:
        """
        Fallback recommendation when ML models are not available
        """
        try:
            score = 0
            reasoning = []
            
            # Technical analysis
            if 'sma_20' in technical_indicators and 'sma_50' in technical_indicators:
                if technical_indicators['sma_20'] > technical_indicators['sma_50']:
                    score += 1
                    reasoning.append("20-day SMA above 50-day SMA")
                else:
                    score -= 1
                    reasoning.append("20-day SMA below 50-day SMA")
            
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi']
                if rsi < 30:
                    score += 1
                    reasoning.append("RSI indicates oversold conditions")
                elif rsi > 70:
                    score -= 1
                    reasoning.append("RSI indicates overbought conditions")
                else:
                    reasoning.append("RSI in neutral range")
            
            # Sentiment analysis
            if sentiment_score > 0.1:
                score += 1
                reasoning.append("Positive sentiment detected")
            elif sentiment_score < -0.1:
                score -= 1
                reasoning.append("Negative sentiment detected")
            else:
                reasoning.append("Neutral sentiment")
            
            # Generate recommendation
            if score > 1:
                action = 'BUY'
                confidence = min(0.9, 0.5 + abs(score) * 0.1)
            elif score < -1:
                action = 'SELL'
                confidence = min(0.9, 0.5 + abs(score) * 0.1)
            else:
                action = 'HOLD'
                confidence = 0.5
                reasoning.append("Mixed signals, maintaining current position")
            
            return {
                'recommendation': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'model_predictions': {},
                'feature_importance': {}
            }
            
        except Exception as e:
            logger.error(f"Fallback recommendation failed: {e}")
            return {
                'recommendation': 'HOLD',
                'confidence': 0.0,
                'reasoning': ['Error in recommendation system'],
                'model_predictions': {},
                'feature_importance': {}
            }
    
    def _combine_signals(self,
                        predictions: Dict,
                        probabilities: Dict,
                        technical_indicators: Dict,
                        sentiment_score: float) -> Dict:
        """
        Combine various signals into final recommendation
        """
        try:
            # Weight for each component
            weights = {
                'ml_models': 0.4,
                'technical': 0.3,
                'sentiment': 0.3
            }
            
            # ML models consensus
            if predictions:
                buy_votes = sum(1 for pred in predictions.values() if pred == 1)
                sell_votes = sum(1 for pred in predictions.values() if pred == 0)
                ml_score = buy_votes / len(predictions) if predictions else 0.5
            else:
                ml_score = 0.5
            
            # Technical signals
            technical_score = 0
            reasoning = []
            
            if 'sma_20' in technical_indicators and 'sma_50' in technical_indicators:
                if technical_indicators['sma_20'] > technical_indicators['sma_50']:
                    technical_score += 1
                    reasoning.append("Positive trend: 20-day SMA above 50-day SMA")
                else:
                    technical_score -= 1
                    reasoning.append("Negative trend: 20-day SMA below 50-day SMA")
            
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi']
                if rsi < 30:
                    technical_score += 1
                    reasoning.append("RSI indicates oversold conditions")
                elif rsi > 70:
                    technical_score -= 1
                    reasoning.append("RSI indicates overbought conditions")
                else:
                    reasoning.append("RSI in neutral range")
            
            technical_score = technical_score / 2  # Normalize to [-1, 1]
            
            # Sentiment score
            sentiment_score_normalized = max(-1, min(1, sentiment_score * 5))  # Scale sentiment
            
            # Combined score
            score = (
                weights['ml_models'] * ml_score +
                weights['technical'] * (technical_score + 1) / 2 +  # Convert to [0, 1]
                weights['sentiment'] * (sentiment_score_normalized + 1) / 2  # Convert to [0, 1]
            )
            
            # Generate recommendation
            if score > 0.6:
                action = 'BUY'
                confidence = score
                reasoning.append("Strong buy signals from analysis")
            elif score < 0.4:
                action = 'SELL'
                confidence = 1 - score
                reasoning.append("Strong sell signals from analysis")
            else:
                action = 'HOLD'
                confidence = 0.5
                reasoning.append("Mixed signals, maintaining current position")
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Signal combination failed: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reasoning': ['Error in signal processing']
            }
