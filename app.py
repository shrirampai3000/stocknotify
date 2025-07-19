# Main Flask application for stock prediction
from flask import Flask, render_template, request, jsonify
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

# Load environment variables
load_dotenv()

app = Flask(__name__)

predictor = StockPredictor()
news_analyzer = NewsAnalyzer()

def get_indian_tickers():
    # List of popular Indian stocks (NSE)
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'SBIN.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS', 'LT.NS',
        'AXISBANK.NS', 'ITC.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS'
    ]

MONITORED_STOCKS = get_indian_tickers()
@app.route('/opportunities')
def opportunities():
    """Classify stocks into Buy, Sell, Hold based on price change and news sentiment"""
    buy_stocks = []
    sell_stocks = []
    hold_stocks = []
    for symbol in MONITORED_STOCKS:
        try:
            stock_data = predictor.get_stock_data(symbol)
            news_sentiment = news_analyzer.get_news_sentiment(symbol)
            prediction = predictor.predict_stock_movement(symbol, news_sentiment)

            price_history = stock_data.get('history', [])
            change_pct = 0
            if len(price_history) >= 2:
                prev = price_history[-2]['close']
                curr = price_history[-1]['close']
                if prev:
                    change_pct = ((curr - prev) / prev) * 100

            stock_info = {
                'symbol': symbol,
                'current_price': stock_data['current_price'],
                'change_pct': round(change_pct, 2),
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'sentiment_score': news_sentiment['overall_sentiment'],
                'news_count': news_sentiment['news_count'],
                'recommendation': prediction['recommendation'],
                'timestamp': datetime.now().isoformat()
            }

            # Improved classification logic for Indian stocks
            recommendation = prediction.get('recommendation', '').lower()
            sentiment = news_sentiment.get('overall_sentiment', 0)
            # Consider a sudden dip (>2% drop) as buy if sentiment is positive
            if recommendation == 'buy' or (change_pct <= -2 and sentiment > 0):
                buy_stocks.append(stock_info)
            # Consider a sudden rise (>2% increase) as sell if sentiment is negative
            elif recommendation == 'sell' or (change_pct >= 2 and sentiment < 0):
                sell_stocks.append(stock_info)
            else:
                hold_stocks.append(stock_info)
        except Exception as e:
            continue
    return render_template('opportunities.html', buy_stocks=buy_stocks, sell_stocks=sell_stocks, hold_stocks=hold_stocks)

@app.route('/')
def index():
    """Main page of the application"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_stock():
    """API endpoint for stock prediction"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        # Get stock data and predictions
        stock_data = predictor.get_stock_data(symbol)
        if stock_data is None:
            return jsonify({'error': f'Could not fetch data for {symbol}'}), 404
        
        # Get news sentiment
        news_sentiment = news_analyzer.get_news_sentiment(symbol)
        
        # Make prediction
        prediction = predictor.predict_stock_movement(symbol, news_sentiment)
        
        return jsonify({
            'symbol': symbol,
            'current_price': stock_data['current_price'],
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'sentiment_score': news_sentiment['overall_sentiment'],
            'news_count': news_sentiment['news_count'],
            'recommendation': prediction['recommendation'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stock/<symbol>')
def stock_details(symbol):
    """Detailed stock analysis page"""
    try:
        stock_data = predictor.get_stock_data(symbol.upper())
        news_sentiment = news_analyzer.get_news_sentiment(symbol.upper())
        prediction = predictor.predict_stock_movement(symbol.upper(), news_sentiment)
        
        return render_template('stock_details.html', 
                             symbol=symbol.upper(),
                             stock_data=stock_data,
                             news_sentiment=news_sentiment,
                             prediction=prediction)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/news/<symbol>')
def get_news(symbol):
    """API endpoint to get news for a stock"""
    try:
        news = news_analyzer.get_news_articles(symbol.upper())
        return jsonify(news)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 