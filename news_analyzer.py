from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
import os
import datetime

class NewsAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.googlenews = GoogleNews(lang='en', period='7d')

    def get_news_sentiment(self, symbol):
        """
        Fetch recent news articles for the stock symbol and calculate overall sentiment score.
        """
        try:
            query = symbol.split('.')[0]
            self.googlenews.clear()
            self.googlenews.search(query)
            articles = self.googlenews.results(sort=True)
            if not articles:
                return {'overall_sentiment': 0, 'news_count': 0}

            sentiment_scores = []
            for article in articles:
                content = article.get('desc', '') or article.get('title', '')
                if content:
                    vader_score = self.analyzer.polarity_scores(content)['compound']
                    tb_score = TextBlob(content).sentiment.polarity
                    avg_score = (vader_score + tb_score) / 2
                    sentiment_scores.append(avg_score)

            if sentiment_scores:
                overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            else:
                overall_sentiment = 0

            return {
                'overall_sentiment': overall_sentiment,
                'news_count': len(articles)
            }
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {e}")
            return {'overall_sentiment': 0, 'news_count': 0}

    def get_news_articles(self, symbol):
        """
        Fetch recent news articles for the stock symbol.
        """
        try:
            today = datetime.datetime.now()
            from_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            query = symbol.split('.')[0]

            all_articles = self.newsapi.get_everything(q=query,
                                                       from_param=from_date,
                                                       to=to_date,
                                                       language='en',
                                                       sort_by='relevancy',
                                                       page_size=50)

            articles = all_articles.get('articles', [])
            simplified_articles = []
            for article in articles:
                simplified_articles.append({
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'url': article.get('url'),
                    'publishedAt': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name')
                })
            return simplified_articles
        except Exception as e:
            print(f"Error fetching news articles for {symbol}: {e}")
            return []
