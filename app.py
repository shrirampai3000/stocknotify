"""
Stock Notification Application - Main Flask Application
"""
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging

# Import application modules
from data_ingestion.market_data import MarketDataFetcher
from data_ingestion.news_sentiment import NewsSentimentAnalyzer
from analysis.market_analysis import MarketAnalyzer
from analysis.recommendation import RecommendationEngine
from config import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_name='default'):
    """Application factory pattern."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize components
    market_data = MarketDataFetcher()
    news_analyzer = NewsSentimentAnalyzer()
    market_analyzer = MarketAnalyzer()
    recommendation_engine = RecommendationEngine()
    
    # Cache for storing analysis results
    analysis_cache = {}
    
    def analyze_stock(symbol: str) -> dict:
        """
        Comprehensive stock analysis including technical, sentiment, and ML predictions
        """
        now = datetime.now()
        
        # Check cache first
        if symbol in analysis_cache:
            last_update, data = analysis_cache[symbol]
            if (now - last_update).seconds < app.config['CACHE_TTL']:
                return data

        try:
            # Fetch market data
            end_date = now
            start_date = end_date - timedelta(days=app.config['HISTORICAL_DAYS'])
            stock_data = market_data.get_stock_data(symbol, start_date, end_date)
            
            if stock_data is None or stock_data.empty or 'Close' not in stock_data.columns:
                logger.error(f"No valid price data for {symbol}")
                return {'symbol': symbol, 'error': 'No valid price data'}

            # Get news and sentiment
            try:
                news_items, sentiment_score = news_analyzer.get_news_sentiment(
                    symbol, 
                    symbol.replace(".NS", "")
                )
            except Exception as e:
                logger.error(f"News sentiment error for {symbol}: {e}")
                news_items, sentiment_score = [], 0.0

            # Technical analysis
            try:
                technical_analysis = market_analyzer.analyze_stock(stock_data, symbol)
            except Exception as e:
                logger.error(f"Technical analysis error for {symbol}: {e}")
                technical_analysis = {
                    'technical_indicators': {}, 
                    'predictions': {}, 
                    'risk_metrics': {}, 
                    'regime': {}
                }

            # Generate recommendations
            try:
                recommendation = recommendation_engine.generate_recommendation(
                    stock_data,
                    technical_analysis.get('technical_indicators', {}),
                    sentiment_score,
                    technical_analysis.get('regime', {})
                )
            except Exception as e:
                logger.error(f"Recommendation error for {symbol}: {e}")
                recommendation = {
                    'recommendation': 'HOLD', 
                    'confidence': 0, 
                    'reasoning': ['Error in recommendation']
                }

            # Prepare response
            try:
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2] if len(stock_data['Close']) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price else 0.0
                prediction = technical_analysis.get('predictions', {}).get('prophet_pred', [current_price])[0]
            except Exception as e:
                logger.error(f"Data formatting error for {symbol}: {e}")
                current_price, change_pct, prediction = 0.0, 0.0, 0.0

            result = {
                'symbol': symbol,
                'current_price': current_price,
                'change_pct': change_pct,
                'technical_analysis': technical_analysis,
                'sentiment_score': sentiment_score,
                'news_count': len(news_items),
                'recent_news': news_items[:5],
                'recommendation': recommendation.get('recommendation', 'HOLD'),
                'confidence': recommendation.get('confidence', 0),
                'prediction': prediction,
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            analysis_cache[symbol] = (now, result)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    @app.route('/')
    def opportunities():
        """
        Enhanced dashboard with comprehensive market analysis
        """
        try:
            # Analyze all stocks
            results = []
            for symbol in app.config['STOCK_LIST']:
                result = analyze_stock(symbol)
                if result and 'error' not in result:
                    results.append(result)

            # Always define all variables for template
            buy_stocks = [r for r in results if r.get('recommendation') == 'BUY'] if results else []
            sell_stocks = [r for r in results if r.get('recommendation') == 'SELL'] if results else []
            hold_stocks = [r for r in results if r.get('recommendation') == 'HOLD'] if results else []
            
            # Fallback: If no buy opportunities, show top gainer as buy
            if not buy_stocks and results:
                top_gainer = max(results, key=lambda x: x.get('change_pct', 0))
                buy_stocks = [top_gainer]
                
            sorted_by_change = sorted(results, key=lambda x: x.get('change_pct', 0)) if results else []
            top_losers = sorted_by_change[:3] if sorted_by_change else []
            top_gainers = sorted_by_change[-3:] if sorted_by_change else []
            
            total_stocks = len(results)
            avg_sentiment = round(np.mean([r.get('sentiment_score', 0) for r in results]), 3) if results else 0.0
            avg_change = round(np.mean([r.get('change_pct', 0) for r in results]), 2) if results else 0.0
            buy_count = len(buy_stocks)
            sell_count = len(sell_stocks)
            hold_count = len(hold_stocks)
            
            # Calculate prediction accuracy
            prediction_results = []
            for symbol in app.config['STOCK_LIST']:
                if symbol in analysis_cache:
                    last_analysis = analysis_cache[symbol][1]
                    if 'prediction' in last_analysis:
                        actual = last_analysis.get('current_price', 0)
                        predicted = last_analysis.get('prediction', 0)
                        prediction_results.append({
                            'symbol': symbol,
                            'predicted': predicted,
                            'actual': actual,
                            'is_correct': abs((predicted - actual) / actual) <= 0.02 if actual else False
                        })
            
            if prediction_results:
                correct_predictions = sum(1 for r in prediction_results if r['is_correct'])
                prediction_accuracy = (correct_predictions / len(prediction_results)) * 100
            else:
                prediction_accuracy = None
                
            return render_template(
                "opportunities.html",
                buy_stocks=sorted(buy_stocks, key=lambda x: x.get('confidence', 0), reverse=True),
                sell_stocks=sorted(sell_stocks, key=lambda x: x.get('confidence', 0), reverse=True),
                hold_stocks=sorted(hold_stocks, key=lambda x: x.get('confidence', 0), reverse=True),
                top_gainers=top_gainers,
                top_losers=top_losers,
                total_stocks=total_stocks,
                avg_sentiment=avg_sentiment,
                avg_change=avg_change,
                buy_count=buy_count,
                sell_count=sell_count,
                hold_count=hold_count,
                prediction_accuracy=round(prediction_accuracy, 1) if prediction_accuracy else None,
                prediction_results=prediction_results,
                last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error in opportunities route: {e}")
            # Always pass all variables, even on error
            return render_template(
                "opportunities.html",
                buy_stocks=[],
                sell_stocks=[],
                hold_stocks=[],
                top_gainers=[],
                top_losers=[],
                total_stocks=0,
                avg_sentiment=0.0,
                avg_change=0.0,
                buy_count=0,
                sell_count=0,
                hold_count=0,
                prediction_accuracy=None,
                prediction_results=[],
                last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                error=str(e)
            )

    @app.route('/api/stock/<symbol>')
    def stock_details(symbol):
        """
        API endpoint for detailed stock analysis
        """
        try:
            result = analyze_stock(symbol)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/health')
    def health_check():
        """
        Health check endpoint
        """
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
