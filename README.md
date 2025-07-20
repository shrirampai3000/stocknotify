# Stock Notification Application

A comprehensive stock market analysis and notification system that provides real-time stock data, technical analysis, sentiment analysis, and trading recommendations.

## 🚀 Features

- **Real-time Stock Data**: Fetch live stock data from Yahoo Finance
- **Technical Analysis**: Calculate RSI, MACD, moving averages, and other technical indicators
- **Sentiment Analysis**: Analyze news sentiment for stocks
- **ML Predictions**: Generate price predictions using multiple models (SARIMA, Prophet, GARCH)
- **Trading Recommendations**: AI-powered buy/sell/hold recommendations
- **Web Dashboard**: Beautiful Flask web interface
- **Risk Metrics**: Value at Risk (VaR), volatility analysis
- **Market Regime Detection**: Identify market conditions using clustering

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for stock data

## 🛠️ Installation

1. **Clone or download the project**
   ```bash
   cd "stock notify"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install additional packages for enhanced functionality**
   ```bash
   # For advanced ML features
   pip install prophet arch ruptures lightgbm xgboost shap
   
   # For additional data sources
   pip install alpha-vantage fredapi
   
   # For news sentiment
   pip install newsapi-python feedparser
   ```

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root for API keys:

```env
# News API for sentiment analysis
NEWS_API_KEY=your_news_api_key

# Alpha Vantage for additional market data
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# FRED API for macroeconomic data
FRED_API_KEY=your_fred_api_key

# Flask secret key
SECRET_KEY=your_secret_key
```

### API Keys (Optional)

- **News API**: Get free API key from [newsapi.org](https://newsapi.org/)
- **Alpha Vantage**: Get free API key from [alphavantage.co](https://alphavantage.co/)
- **FRED API**: Get free API key from [fred.stlouisfed.org](https://fred.stlouisfed.org/)

## 🚀 Usage

### Quick Start

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **View the dashboard**
   - See stock opportunities
   - View technical analysis
   - Check sentiment scores
   - Get trading recommendations

### Testing the Application

Run the test suite to verify everything works:

```bash
python test_app.py
```

### Command Line Usage

```bash
# Test specific functionality
python -c "from app import app; print('App loaded successfully')"

# Run with debug mode
python app.py
```

## 📊 Dashboard Features

### Main Dashboard (`/`)
- **Buy Opportunities**: Stocks with strong buy signals
- **Sell Signals**: Stocks to consider selling
- **Hold Recommendations**: Stocks to maintain positions
- **Top Gainers/Losers**: Best and worst performing stocks
- **Market Sentiment**: Overall market sentiment score
- **Prediction Accuracy**: ML model performance metrics

### API Endpoints

- `GET /`: Main dashboard
- `GET /api/stock/<symbol>`: Detailed analysis for specific stock

## 🏗️ Project Structure

```
stock notify/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── test_app.py           # Test suite
├── README.md             # This file
├── data_ingestion/       # Data fetching modules
│   ├── market_data.py    # Stock data fetcher
│   └── news_sentiment.py # News and sentiment analysis
├── analysis/             # Analysis modules
│   ├── market_analysis.py    # Technical analysis
│   └── recommendation.py     # ML recommendations
├── templates/            # HTML templates
│   ├── index.html        # Main dashboard
│   ├── opportunities.html # Opportunities page
│   └── ...               # Other pages
└── static/               # Static files
    └── style.css         # CSS styles
```

## 🔍 How It Works

### 1. Data Ingestion
- Fetches stock data from Yahoo Finance
- Collects news articles and sentiment
- Handles multiple data sources with fallbacks

### 2. Technical Analysis
- Calculates technical indicators (RSI, MACD, SMA)
- Generates price predictions using ML models
- Computes risk metrics (VaR, volatility)

### 3. Sentiment Analysis
- Analyzes news sentiment using TextBlob
- Processes RSS feeds for additional news
- Calculates overall sentiment scores

### 4. Recommendation Engine
- Combines technical, sentiment, and ML signals
- Generates buy/sell/hold recommendations
- Provides confidence scores and reasoning

## 🛡️ Error Handling

The application is designed to be robust:

- **Graceful degradation**: Works even if some dependencies are missing
- **Fallback mechanisms**: Uses alternative data sources when primary fails
- **Comprehensive logging**: Detailed error messages for debugging
- **Data validation**: Checks for empty or invalid data

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **No Stock Data**
   - Check internet connection
   - Verify stock symbols are correct
   - Try different stock symbols

3. **Missing Dependencies**
   - Install optional packages for enhanced features
   - Application works with basic dependencies

4. **API Rate Limits**
   - Some APIs have rate limits
   - Application uses caching to minimize API calls

### Debug Mode

Run with debug mode for detailed error messages:

```bash
python app.py
```

## 📈 Stock List

The application tracks these Indian stocks by default:
- RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS
- HINDUNILVR.NS, HDFC.NS, SBIN.NS, BAJFINANCE.NS, BHARTIARTL.NS

You can modify the `STOCK_LIST` in `app.py` to track different stocks.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It does not constitute financial advice. Always do your own research and consult with financial professionals before making investment decisions.

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Run the test suite: `python test_app.py`
3. Check the logs for error messages
4. Verify all dependencies are installed

---

**Happy Trading! 📈**