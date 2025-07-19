# Stock Notify - Advanced Stock Analysis & Portfolio Management

Stock Notify is a comprehensive stock analysis and portfolio management application that helps investors make informed decisions based on technical analysis, sentiment analysis, and portfolio tracking.

## Features

- **Multi-Market Analysis**: Track stocks from both Indian (NSE) and US markets
- **Technical Analysis**: Advanced technical indicators including RSI, MACD, Bollinger Bands, ATR, OBV, and support/resistance levels
- **Sentiment Analysis**: Combined news and social media sentiment analysis
- **Portfolio Management**: Track your investments, performance, and allocation
- **Automated Alerts**: Get notified about significant price movements and sentiment changes
- **Performance Tracking**: Monitor the accuracy of predictions over time
- **Scheduled Reports**: Receive daily portfolio summaries and weekly performance reports

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file:
   ```
   NEWSAPI_KEY=your_news_api_key
   TWITTER_CONSUMER_KEY=your_twitter_consumer_key
   TWITTER_CONSUMER_SECRET=your_twitter_consumer_secret
   TWITTER_ACCESS_TOKEN=your_twitter_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   SECRET_KEY=your_flask_secret_key
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Use the navigation menu to access different features:
   - **Opportunities**: View buy/sell recommendations based on technical and sentiment analysis
   - **Portfolio**: Manage your investment portfolio and track performance
   - **Performance**: Monitor prediction accuracy and model performance
   - **Settings**: Configure notification preferences

## Components

- **Stock Predictor**: Technical analysis and price prediction
- **News Analyzer**: News sentiment analysis
- **Social Sentiment Analyzer**: Social media sentiment analysis
- **Portfolio Tracker**: Investment portfolio management
- **Performance Tracker**: Prediction accuracy tracking
- **Notification System**: Automated alerts and reports
- **Task Scheduler**: Automated background tasks

## Configuration

- **Portfolio**: Edit `portfolio.json` to customize your portfolio
- **Notifications**: Configure notification settings in the Settings page
- **Monitored Stocks**: Modify the stock lists in `app.py` to track different stocks

## License

MIT License

## Disclaimer

This application is for informational purposes only and does not constitute financial advice. Always do your own research before making investment decisions.