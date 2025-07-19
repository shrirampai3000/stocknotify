import os
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

class SocialSentimentAnalyzer:
    def __init__(self, use_mock_data=False):
        self.use_mock_data = use_mock_data
        self.analyzer = SentimentIntensityAnalyzer()
        
        if not use_mock_data:
            consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
            consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
            access_token = os.getenv('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

            if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
                print("Warning: Twitter API credentials not fully set, using mock data instead")
                self.use_mock_data = True
            else:
                try:
                    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
                    self.api = tweepy.API(auth)
                    # Test the connection
                    self.api.verify_credentials()
                except Exception as e:
                    print(f"Warning: Could not authenticate with Twitter API: {e}")
                    self.use_mock_data = True

    def get_twitter_sentiment(self, query, count=100):
        """
        Fetch recent tweets matching the query and calculate average sentiment score.
        If use_mock_data is True, returns mock sentiment data.
        """
        if self.use_mock_data:
            return self._get_mock_sentiment(query)
            
        try:
            # Updated for newer tweepy versions
            try:
                # Try the newer API first
                tweets = tweepy.Cursor(self.api.search_tweets, q=query, lang='en', tweet_mode='extended').items(count)
            except AttributeError:
                # Fall back to older API
                tweets = tweepy.Cursor(self.api.search, q=query, lang='en', tweet_mode='extended').items(count)
                
            sentiment_scores = []
            tweet_texts = []
            
            for tweet in tweets:
                try:
                    text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
                    tweet_texts.append(text)
                    score = self.analyzer.polarity_scores(text)['compound']
                    sentiment_scores.append(score)
                except Exception as e:
                    print(f"Error processing tweet: {e}")
                    continue

            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            else:
                avg_sentiment = 0

            return {
                'average_sentiment': avg_sentiment,
                'tweet_count': len(sentiment_scores),
                'tweets': tweet_texts
            }
        except Exception as e:
            print(f"Error fetching Twitter sentiment for query '{query}': {e}")
            return self._get_mock_sentiment(query)
            
    def _get_mock_sentiment(self, query):
        """
        Generate mock sentiment data for testing when Twitter API is unavailable
        """
        # Generate a sentiment score that's somewhat related to the query
        # This is just for demonstration purposes
        positive_companies = ['apple', 'microsoft', 'amazon', 'google', 'tesla']
        negative_companies = ['oil', 'coal', 'gas']
        
        base_sentiment = 0
        query_lower = query.lower()
        
        # Bias sentiment based on company name
        for company in positive_companies:
            if company in query_lower:
                base_sentiment += 0.3
                break
                
        for company in negative_companies:
            if company in query_lower:
                base_sentiment -= 0.2
                break
        
        # Add some randomness
        sentiment = base_sentiment + random.uniform(-0.3, 0.3)
        sentiment = max(-1.0, min(1.0, sentiment))  # Clamp between -1 and 1
        
        # Generate mock tweet count
        tweet_count = random.randint(5, 30)
        
        # Generate mock tweets
        mock_tweets = [
            f"I think {query} is a great company! #stocks",
            f"Just bought some shares of {query}. Looking promising.",
            f"Market analysis suggests {query} might be trending upward.",
            f"Not sure about {query}'s latest product announcement.",
            f"Earnings report for {query} was better than expected!"
        ]
        
        return {
            'average_sentiment': sentiment,
            'tweet_count': tweet_count,
            'tweets': mock_tweets[:random.randint(1, 5)]
        }
