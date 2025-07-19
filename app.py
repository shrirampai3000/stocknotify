# Main Flask application for stock prediction
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv
from stock_predictor import StockPredictor
from news_analyzer import NewsAnalyzer
# Import with error handling for social sentiment
try:
    from social_sentiment import SocialSentimentAnalyzer
    social_sentiment_available = True
except Exception as e:
    print(f"Warning: Social sentiment analysis not available: {e}")
    social_sentiment_available = False
from portfolio_tracker import PortfolioTracker
from notification_system import NotificationSystem
from performance_tracker import PerformanceTracker
from scheduler import start_scheduler

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'stocknotify-secret-key')

# Initialize components
predictor = StockPredictor()
news_analyzer = NewsAnalyzer()

# Initialize social sentiment analyzer with error handling
if social_sentiment_available:
    try:
        # Use mock data to avoid Twitter API issues
        social_sentiment_analyzer = SocialSentimentAnalyzer(use_mock_data=True)
        print("Using mock data for social sentiment analysis")
    except Exception as e:
        print(f"Warning: Could not initialize social sentiment analyzer: {e}")
        social_sentiment_available = False
else:
    social_sentiment_analyzer = None

# Initialize other components
portfolio_tracker = PortfolioTracker()
notification_system = NotificationSystem()
performance_tracker = PerformanceTracker()

# Simple in-memory cache for news sentiment and articles
news_cache = {}
NEWS_CACHE_TTL = 1800  # 30 minutes

# Start the scheduler (commented out for initial testing)
try:
    # start_scheduler()  # Uncomment this line after ensuring the app runs correctly
    print("Scheduler disabled for initial testing")
except Exception as e:
    print(f"Error with scheduler: {e}")

def get_indian_tickers():
    # List of popular Indian stocks (NSE)
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'SBIN.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS', 'LT.NS',
        'AXISBANK.NS', 'ITC.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS'
    ]

def get_us_tickers():
    # List of popular US stocks
    return [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META',
        'TSLA', 'NVDA', 'JPM', 'V', 'WMT'
    ]

# Get stocks based on market selection
def get_monitored_stocks(market='all'):
    if market == 'india':
        return get_indian_tickers()
    elif market == 'us':
        return get_us_tickers()
    else:  # 'all'
        return get_indian_tickers() + get_us_tickers()

MONITORED_STOCKS = get_monitored_stocks()

# Portfolio tracking
portfolio_data = {
    'RELIANCE.NS': {'shares': 10, 'buy_price': 2500},
    'AAPL': {'shares': 5, 'buy_price': 180},
    'MSFT': {'shares': 3, 'buy_price': 350}
}
@app.route('/opportunities')
@app.route('/')
def opportunities():
    market = request.args.get('market', 'all')
    stocks_to_monitor = get_monitored_stocks(market)
    """Classify stocks into Buy, Sell, Hold based on price change and news sentiment"""
    buy_stocks = []
    sell_stocks = []
    hold_stocks = []
    all_stocks = []
    prediction_results = []
    correct_predictions = 0
    total_predictions = 0
    import time
    now = time.time()
    for symbol in stocks_to_monitor:
        try:
            stock_data = predictor.get_stock_data(symbol)
            # Caching for news sentiment and articles
            cache_entry = news_cache.get(symbol)
            if cache_entry and now - cache_entry['timestamp'] < NEWS_CACHE_TTL:
                news_sentiment = cache_entry['news_sentiment']
                recent_news = cache_entry['recent_news']
            else:
                news_sentiment = news_analyzer.get_news_sentiment(symbol)
                recent_news = news_analyzer.get_news_articles(symbol) if hasattr(news_analyzer, 'get_news_articles') else []
                news_cache[symbol] = {
                    'news_sentiment': news_sentiment,
                    'recent_news': recent_news,
                    'timestamp': now
                }
            # Get social media sentiment if available
            social_sentiment = {'overall_sentiment': 0, 'tweet_count': 0}
            if social_sentiment_available and social_sentiment_analyzer:
                try:
                    # Extract company name from ticker for better social media search
                    company_name = symbol.split('.')[0]
                    social_sentiment = social_sentiment_analyzer.get_twitter_sentiment(company_name)
                except Exception as e:
                    print(f"Error getting social sentiment for {symbol}: {e}")
                
            # Combine news and social sentiment with weighted average
            news_weight = 0.7
            social_weight = 0.3
            news_sentiment_value = news_sentiment['overall_sentiment']
            social_sentiment_value = social_sentiment.get('average_sentiment', 0)
            
            # Calculate weighted sentiment - prioritize news sentiment if social is unavailable
            if social_sentiment_available and news_sentiment['news_count'] > 0 and social_sentiment.get('tweet_count', 0) > 0:
                combined_sentiment = (news_sentiment_value * news_weight) + (social_sentiment_value * social_weight)
            elif news_sentiment['news_count'] > 0:
                combined_sentiment = news_sentiment_value
            elif social_sentiment_available and social_sentiment.get('tweet_count', 0) > 0:
                combined_sentiment = social_sentiment_value
            else:
                combined_sentiment = 0
                
            combined_sentiment_data = {
                'overall_sentiment': combined_sentiment,
                'news_count': news_sentiment['news_count'],
                'tweet_count': social_sentiment.get('tweet_count', 0)
            }
            prediction = predictor.predict_stock_movement(symbol, combined_sentiment_data)

            price_history = stock_data.get('history', [])
            change_pct = 0
            if len(price_history) >= 2:
                prev = price_history[-2]['close']
                curr = price_history[-1]['close']
                if prev:
                    change_pct = ((curr - prev) / prev) * 100

            # Validation: compare previous prediction with actual movement
            actual_movement = None
            if len(price_history) >= 2:
                if curr > prev:
                    actual_movement = 'buy'
                elif curr < prev:
                    actual_movement = 'sell'
                else:
                    actual_movement = 'hold'
            predicted_movement = prediction.get('recommendation', '').lower()
            is_correct = actual_movement == predicted_movement if actual_movement else None
            if is_correct is not None:
                total_predictions += 1
                if is_correct:
                    correct_predictions += 1

            stock_info = {
                'symbol': symbol,
                'current_price': stock_data['current_price'],
                'change_pct': round(change_pct, 2),
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'sentiment_score': combined_sentiment,
                'news_count': combined_sentiment_data['news_count'],
                'recommendation': prediction['recommendation'],
                'timestamp': datetime.now().isoformat(),
                'recent_news': recent_news,
                'actual_movement': actual_movement,
                'is_correct': is_correct
            }

            # Improved classification logic for Indian stocks
            sentiment = combined_sentiment
            if predicted_movement == 'buy' or (change_pct <= -2 and sentiment > 0):
                buy_stocks.append(stock_info)
            elif predicted_movement == 'sell' or (change_pct >= 2 and sentiment < 0):
                sell_stocks.append(stock_info)
            else:
                hold_stocks.append(stock_info)
            all_stocks.append(stock_info)
            prediction_results.append({'symbol': symbol, 'predicted': predicted_movement, 'actual': actual_movement, 'is_correct': is_correct})
        except Exception as e:
            continue

    # Holistic market stats
    total_stocks = len(all_stocks)
    avg_sentiment = round(np.mean([s['sentiment_score'] for s in all_stocks]), 3) if all_stocks else 0
    avg_change = round(np.mean([s['change_pct'] for s in all_stocks]), 2) if all_stocks else 0
    buy_count = len(buy_stocks)
    sell_count = len(sell_stocks)
    hold_count = len(hold_stocks)
    top_gainers = sorted(all_stocks, key=lambda x: x['change_pct'], reverse=True)[:5]
    top_losers = sorted(all_stocks, key=lambda x: x['change_pct'])[:5]
    prediction_accuracy = round((correct_predictions / total_predictions) * 100, 2) if total_predictions > 0 else None

    # Calculate portfolio performance
    portfolio_value = 0
    portfolio_cost = 0
    portfolio_performance = []
    
    for symbol, details in portfolio_data.items():
        stock_info = next((s for s in all_stocks if s['symbol'] == symbol), None)
        if stock_info:
            current_value = stock_info['current_price'] * details['shares']
            cost_basis = details['buy_price'] * details['shares']
            profit_loss = current_value - cost_basis
            profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            portfolio_value += current_value
            portfolio_cost += cost_basis
            
            portfolio_performance.append({
                'symbol': symbol,
                'shares': details['shares'],
                'buy_price': details['buy_price'],
                'current_price': stock_info['current_price'],
                'current_value': current_value,
                'profit_loss': profit_loss,
                'profit_loss_pct': round(profit_loss_pct, 2),
                'recommendation': stock_info['recommendation']
            })
    
    total_profit_loss = portfolio_value - portfolio_cost
    total_profit_loss_pct = (total_profit_loss / portfolio_cost) * 100 if portfolio_cost > 0 else 0
    
    # Get last updated time
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('opportunities.html',
        buy_stocks=buy_stocks,
        sell_stocks=sell_stocks,
        hold_stocks=hold_stocks,
        total_stocks=total_stocks,
        avg_sentiment=avg_sentiment,
        avg_change=avg_change,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
        top_gainers=top_gainers,
        top_losers=top_losers,
        prediction_results=prediction_results,
        prediction_accuracy=prediction_accuracy,
        portfolio_performance=portfolio_performance,
        portfolio_value=round(portfolio_value, 2),
        portfolio_cost=round(portfolio_cost, 2),
        total_profit_loss=round(total_profit_loss, 2),
        total_profit_loss_pct=round(total_profit_loss_pct, 2),
        last_updated=last_updated,
        selected_market=market
    )

# API endpoints for notifications and alerts
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get stock alerts based on significant price or sentiment changes"""
    threshold = float(request.args.get('threshold', 5.0))  # Default 5% change
    alerts = []
    
    for symbol in MONITORED_STOCKS:
        try:
            stock_data = predictor.get_stock_data(symbol)
            if not stock_data:
                continue
                
            price_change = stock_data.get('price_change', 0)
            
            # Check if stock is in portfolio
            in_portfolio = symbol in portfolio_data
            
            # Generate alerts based on significant changes
            if abs(price_change) >= threshold:
                alert_type = 'price_change'
                direction = 'up' if price_change > 0 else 'down'
                priority = 'high' if in_portfolio else 'medium'
                
                alerts.append({
                    'symbol': symbol,
                    'type': alert_type,
                    'message': f"{symbol} has moved {direction} by {abs(price_change)}%",
                    'timestamp': datetime.now().isoformat(),
                    'priority': priority
                })
                
            # Get cached sentiment if available
            cache_entry = news_cache.get(symbol)
            if cache_entry:
                sentiment = cache_entry['news_sentiment']['overall_sentiment']
                
                # Alert on extreme sentiment
                if abs(sentiment) >= 0.5:
                    sentiment_direction = 'positive' if sentiment > 0 else 'negative'
                    priority = 'high' if in_portfolio else 'medium'
                    
                    alerts.append({
                        'symbol': symbol,
                        'type': 'sentiment_change',
                        'message': f"{symbol} has {sentiment_direction} sentiment score of {sentiment}",
                        'timestamp': datetime.now().isoformat(),
                        'priority': priority
                    })
        except Exception as e:
            continue
            
    return jsonify(alerts)

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio performance"""
    portfolio_data = []
    total_value = 0
    total_cost = 0
    
    for symbol, details in portfolio_data.items():
        try:
            stock_data = predictor.get_stock_data(symbol)
            if not stock_data:
                continue
                
            current_price = stock_data['current_price']
            current_value = current_price * details['shares']
            cost_basis = details['buy_price'] * details['shares']
            profit_loss = current_value - cost_basis
            profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            total_value += current_value
            total_cost += cost_basis
            
            portfolio_data.append({
                'symbol': symbol,
                'shares': details['shares'],
                'buy_price': details['buy_price'],
                'current_price': current_price,
                'current_value': round(current_value, 2),
                'profit_loss': round(profit_loss, 2),
                'profit_loss_pct': round(profit_loss_pct, 2)
            })
        except Exception as e:
            continue
    
    return jsonify({
        'portfolio': portfolio_data,
        'total_value': round(total_value, 2),
        'total_cost': round(total_cost, 2),
        'total_profit_loss': round(total_value - total_cost, 2),
        'total_profit_loss_pct': round(((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0, 2)
    })

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get stock recommendations based on technical analysis and sentiment"""
    market = request.args.get('market', 'all')
    stocks = get_monitored_stocks(market)
    recommendations = []
    
    for symbol in stocks:
        try:
            stock_data = predictor.get_stock_data(symbol)
            if not stock_data:
                continue
                
            # Get cached sentiment
            cache_entry = news_cache.get(symbol)
            if not cache_entry:
                continue
                
            sentiment_data = cache_entry['news_sentiment']
            prediction = predictor.predict_stock_movement(symbol, sentiment_data)
            
            recommendations.append({
                'symbol': symbol,
                'current_price': stock_data['current_price'],
                'recommendation': prediction['recommendation'],
                'confidence': prediction['confidence'],
                'sentiment': sentiment_data['overall_sentiment']
            })
        except Exception as e:
            continue
            
    return jsonify(recommendations)

@app.route('/portfolio')
def portfolio():
    """Display portfolio performance"""
    portfolio_data = portfolio_tracker.get_portfolio_performance()
    performance_history = portfolio_tracker.track_performance_history(days=30)
    allocation = portfolio_tracker.get_portfolio_allocation()
    
    return render_template('portfolio.html',
        portfolio=portfolio_data['positions'],
        total_value=portfolio_data['total_value'],
        total_cost=portfolio_data['total_cost'],
        total_profit_loss=portfolio_data['total_profit_loss'],
        total_profit_loss_pct=portfolio_data['total_profit_loss_pct'],
        performance_history=performance_history,
        allocation=allocation
    )

@app.route('/portfolio/add', methods=['POST'])
def add_position():
    """Add a new position to portfolio"""
    symbol = request.form.get('symbol')
    shares = float(request.form.get('shares'))
    buy_price = float(request.form.get('buy_price'))
    buy_date = request.form.get('buy_date')
    
    if portfolio_tracker.add_position(symbol, shares, buy_price, buy_date):
        flash(f"Added {shares} shares of {symbol} at ${buy_price}", "success")
    else:
        flash("Error adding position", "danger")
        
    return redirect(url_for('portfolio'))

@app.route('/portfolio/remove', methods=['POST'])
def remove_position():
    """Remove a position from portfolio"""
    symbol = request.form.get('symbol')
    shares = request.form.get('shares')
    
    if shares:
        shares = float(shares)
    else:
        shares = None
        
    if portfolio_tracker.remove_position(symbol, shares):
        flash(f"Removed position for {symbol}", "success")
    else:
        flash("Error removing position", "danger")
        
    return redirect(url_for('portfolio'))

@app.route('/performance')
def performance():
    """Display prediction performance metrics"""
    accuracy_metrics = performance_tracker.get_accuracy_metrics()
    accuracy_by_symbol = performance_tracker.get_accuracy_by_symbol()
    accuracy_chart = performance_tracker.generate_accuracy_chart()
    
    return render_template('performance.html',
        accuracy_metrics=accuracy_metrics,
        accuracy_by_symbol=accuracy_by_symbol,
        accuracy_chart=accuracy_chart
    )

@app.route('/settings')
def settings():
    """Display and update notification settings"""
    config = notification_system.config
    
    return render_template('settings.html',
        config=config
    )

@app.route('/settings/update', methods=['POST'])
def update_settings():
    """Update notification settings"""
    # Update email settings
    notification_system.config['email']['enabled'] = 'email_enabled' in request.form
    notification_system.config['email']['smtp_server'] = request.form.get('smtp_server')
    notification_system.config['email']['smtp_port'] = int(request.form.get('smtp_port'))
    notification_system.config['email']['username'] = request.form.get('email_username')
    
    # Only update password if provided
    if request.form.get('email_password'):
        notification_system.config['email']['password'] = request.form.get('email_password')
        
    # Update recipients
    recipients = request.form.get('email_recipients', '')
    notification_system.config['email']['recipients'] = [r.strip() for r in recipients.split(',')] if recipients else []
    
    # Update thresholds
    notification_system.config['thresholds']['price_change'] = float(request.form.get('price_threshold'))
    notification_system.config['thresholds']['sentiment_change'] = float(request.form.get('sentiment_threshold'))
    
    # Update notification types
    notification_system.config['notification_types']['price_alerts'] = 'price_alerts' in request.form
    notification_system.config['notification_types']['sentiment_alerts'] = 'sentiment_alerts' in request.form
    notification_system.config['notification_types']['portfolio_alerts'] = 'portfolio_alerts' in request.form
    notification_system.config['notification_types']['daily_summary'] = 'daily_summary' in request.form
    
    # Save config
    if notification_system.save_config():
        flash("Settings updated successfully", "success")
    else:
        flash("Error updating settings", "danger")
        
    return redirect(url_for('settings'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)