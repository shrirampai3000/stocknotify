"""
Technical and statistical analysis module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available. Install with: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("ARCH not available. Install with: pip install arch")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Install with: pip install networkx")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logger.warning("Ruptures not available. Install with: pip install ruptures")

class MarketAnalyzer:
    def __init__(self):
        self.models = {}
        
    def analyze_stock(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Comprehensive stock analysis including technical indicators,
        predictions, and risk metrics
        """
        try:
            results = {
                'technical_indicators': self._calculate_technical_indicators(data),
                'predictions': self._generate_predictions(data),
                'risk_metrics': self._calculate_risk_metrics(data),
                'regime': self._detect_market_regime(data)
            }
            return results
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return {
                'technical_indicators': {},
                'predictions': {},
                'risk_metrics': {},
                'regime': {}
            }
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate common technical indicators
        """
        try:
            if data.empty or 'Close' not in data.columns:
                return {}
                
            close = data['Close']
            
            # Moving averages
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return {
                'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else 0.0,
                'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else 0.0,
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
                'macd_signal': float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0.0
            }
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {}
    
    def _generate_predictions(self, data: pd.DataFrame) -> Dict:
        """
        Generate price predictions using multiple models
        """
        try:
            if data.empty or 'Close' not in data.columns:
                return {}
                
            predictions = {}
            current_price = data['Close'].iloc[-1]
            
            # SARIMA prediction
            if STATSMODELS_AVAILABLE and len(data) > 30:
                try:
                    model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
                    results = model.fit(disp=False)
                    sarima_pred = results.forecast(steps=5)
                    predictions['sarima_pred'] = sarima_pred.tolist()
                except Exception as e:
                    logger.warning(f"SARIMA prediction failed: {e}")
                    predictions['sarima_pred'] = [current_price] * 5
            else:
                predictions['sarima_pred'] = [current_price] * 5
            
            # Prophet prediction
            if PROPHET_AVAILABLE and len(data) > 30:
                try:
                    df_prophet = pd.DataFrame({'ds': data.index, 'y': data['Close']})
                    prophet = Prophet(daily_seasonality=True)
                    prophet.fit(df_prophet)
                    future = prophet.make_future_dataframe(periods=5)
                    prophet_forecast = prophet.predict(future)
                    predictions['prophet_pred'] = prophet_forecast['yhat'].tail(5).tolist()
                except Exception as e:
                    logger.warning(f"Prophet prediction failed: {e}")
                    predictions['prophet_pred'] = [current_price] * 5
            else:
                predictions['prophet_pred'] = [current_price] * 5
            
            # GARCH volatility forecast
            if ARCH_AVAILABLE and len(data) > 30:
                try:
                    returns = 100 * data['Close'].pct_change().dropna()
                    if len(returns) > 10:
                        garch = arch_model(returns, vol='Garch', p=1, q=1)
                        garch_fit = garch.fit(disp='off')
                        garch_forecast = garch_fit.forecast(horizon=5)
                        predictions['volatility_forecast'] = garch_forecast.variance.values[-1].tolist()
                    else:
                        predictions['volatility_forecast'] = [0.01] * 5
                except Exception as e:
                    logger.warning(f"GARCH prediction failed: {e}")
                    predictions['volatility_forecast'] = [0.01] * 5
            else:
                predictions['volatility_forecast'] = [0.01] * 5
            
            return predictions
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return {'sarima_pred': [0.0] * 5, 'prophet_pred': [0.0] * 5, 'volatility_forecast': [0.01] * 5}
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate various risk metrics
        """
        try:
            if data.empty or 'Close' not in data.columns:
                return {}
                
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) == 0:
                return {}
            
            # Value at Risk
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95) if not pd.isna(cvar_95) else 0.0,
                'annual_volatility': float(volatility) if not pd.isna(volatility) else 0.0
            }
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect market regime using clustering and change point detection
        """
        try:
            if data.empty or 'Close' not in data.columns:
                return {}
                
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < 20:
                return {'regimes': [-1] * len(returns), 'change_points': []}
            
            # Prepare features
            volatility = returns.rolling(window=20).std()
            features = pd.concat([returns, volatility], axis=1).dropna()
            
            if len(features) < 5:
                return {'regimes': [-1] * len(returns), 'change_points': []}
            
            # Regime detection using DBSCAN
            if SKLEARN_AVAILABLE:
                try:
                    clustering = DBSCAN(eps=0.3, min_samples=5)
                    labels = clustering.fit_predict(features)
                    regimes = labels.tolist()
                except Exception as e:
                    logger.warning(f"DBSCAN clustering failed: {e}")
                    regimes = [-1] * len(features)
            else:
                regimes = [-1] * len(features)
            
            # Change point detection
            if RUPTURES_AVAILABLE and len(returns) > 10:
                try:
                    algo = rpt.Pelt(model="rbf").fit(returns.values.reshape(-1, 1))
                    change_points = algo.predict(pen=10)
                except Exception as e:
                    logger.warning(f"Change point detection failed: {e}")
                    change_points = []
            else:
                change_points = []
            
            return {
                'regimes': regimes,
                'change_points': change_points
            }
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {'regimes': [-1], 'change_points': []}
