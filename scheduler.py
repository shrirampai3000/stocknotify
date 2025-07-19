from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import os
import json
from portfolio_tracker import PortfolioTracker
from notification_system import NotificationSystem
from performance_tracker import PerformanceTracker
from stock_predictor import StockPredictor
from news_analyzer import NewsAnalyzer

class TaskScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.portfolio_tracker = PortfolioTracker()
        self.notification_system = NotificationSystem()
        self.performance_tracker = PerformanceTracker()
        self.stock_predictor = StockPredictor()
        self.news_analyzer = NewsAnalyzer()
        
    def start(self):
        """Start the scheduler"""
        # Daily portfolio summary (after market close)
        self.scheduler.add_job(
            self.send_portfolio_summary,
            CronTrigger(hour=16, minute=30, day_of_week='mon-fri'),
            id='portfolio_summary'
        )
        
        # Validate predictions (every 4 hours during market hours)
        self.scheduler.add_job(
            self.validate_predictions,
            CronTrigger(hour='9-16/4', minute=0, day_of_week='mon-fri'),
            id='validate_predictions'
        )
        
        # Check for alerts (every 30 minutes during market hours)
        self.scheduler.add_job(
            self.check_alerts,
            CronTrigger(minute='*/30', hour='9-16', day_of_week='mon-fri'),
            id='check_alerts'
        )
        
        # Weekly performance report (Sunday evening)
        self.scheduler.add_job(
            self.send_weekly_report,
            CronTrigger(day_of_week='sun', hour=18, minute=0),
            id='weekly_report'
        )
        
        # Start the scheduler
        self.scheduler.start()
        print(f"Scheduler started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        print(f"Scheduler stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def send_portfolio_summary(self):
        """Send daily portfolio summary"""
        try:
            print(f"Generating portfolio summary at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            portfolio_data = self.portfolio_tracker.get_portfolio_performance()
            self.notification_system.generate_portfolio_summary(portfolio_data)
            print(f"Portfolio summary sent successfully")
        except Exception as e:
            print(f"Error sending portfolio summary: {e}")
            
    def validate_predictions(self):
        """Validate previous predictions"""
        try:
            print(f"Validating predictions at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            portfolio = self.portfolio_tracker.portfolio
            
            # Validate predictions for portfolio stocks first
            for symbol in portfolio:
                try:
                    stock_data = self.stock_predictor.get_stock_data(symbol)
                    if stock_data:
                        current_price = stock_data['current_price']
                        validated = self.performance_tracker.validate_prediction(symbol, current_price)
                        print(f"Validated {validated} predictions for {symbol}")
                except Exception as e:
                    print(f"Error validating predictions for {symbol}: {e}")
                    continue
                    
            print(f"Prediction validation completed")
        except Exception as e:
            print(f"Error in prediction validation: {e}")
            
    def check_alerts(self):
        """Check for stock alerts"""
        try:
            print(f"Checking for alerts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            portfolio = self.portfolio_tracker.portfolio
            
            # Get data for portfolio stocks
            stock_data_list = []
            for symbol in portfolio:
                try:
                    stock_data = self.stock_predictor.get_stock_data(symbol)
                    if stock_data:
                        # Get sentiment
                        news_sentiment = self.news_analyzer.get_news_sentiment(symbol)
                        
                        stock_data_list.append({
                            'symbol': symbol,
                            'current_price': stock_data['current_price'],
                            'change_pct': stock_data.get('price_change', 0),
                            'sentiment_score': news_sentiment['overall_sentiment'],
                            'recommendation': 'BUY' if stock_data.get('price_change', 0) > 0 and news_sentiment['overall_sentiment'] > 0 else 'SELL'
                        })
                except Exception as e:
                    print(f"Error getting data for {symbol}: {e}")
                    continue
                    
            # Send alerts
            alerts_sent = self.notification_system.check_and_send_alerts(stock_data_list, portfolio)
            print(f"Sent {alerts_sent} alerts")
        except Exception as e:
            print(f"Error checking alerts: {e}")
            
    def send_weekly_report(self):
        """Send weekly performance report"""
        try:
            print(f"Generating weekly report at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get portfolio performance
            portfolio_data = self.portfolio_tracker.get_portfolio_performance()
            
            # Get prediction accuracy metrics
            accuracy_metrics = self.performance_tracker.get_accuracy_metrics(days=7)
            accuracy_by_symbol = self.performance_tracker.get_accuracy_by_symbol(days=7)
            
            # Create report message
            subject = f"Weekly Stock Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            message = f"""
            <html>
            <body>
                <h2>Weekly Stock Performance Report</h2>
                
                <h3>Portfolio Summary</h3>
                <p><strong>Total Value:</strong> ${portfolio_data['total_value']}</p>
                <p><strong>Weekly Change:</strong> ${portfolio_data['total_profit_loss']} ({portfolio_data['total_profit_loss_pct']}%)</p>
                
                <h3>Prediction Accuracy (Last 7 Days)</h3>
                <p><strong>Overall Accuracy:</strong> {accuracy_metrics.get('overall_accuracy', 'N/A')}%</p>
                <p><strong>Buy Accuracy:</strong> {accuracy_metrics.get('rise_accuracy', 'N/A')}%</p>
                <p><strong>Sell Accuracy:</strong> {accuracy_metrics.get('fall_accuracy', 'N/A')}%</p>
                
                <h3>Top Performing Stocks</h3>
                <table border="1" cellpadding="5">
                    <tr>
                        <th>Symbol</th>
                        <th>Weekly Change (%)</th>
                    </tr>
            """
            
            # Add top performing stocks
            top_performers = sorted(portfolio_data['positions'], key=lambda x: x['profit_loss_pct'], reverse=True)[:5]
            for stock in top_performers:
                message += f"""
                    <tr>
                        <td>{stock['symbol']}</td>
                        <td>{stock['profit_loss_pct']}%</td>
                    </tr>
                """
                
            message += """
                </table>
            </body>
            </html>
            """
            
            # Send report
            self.notification_system.send_email_notification(subject, message)
            print(f"Weekly report sent successfully")
        except Exception as e:
            print(f"Error sending weekly report: {e}")
            
# Create scheduler instance
scheduler = TaskScheduler()

# Function to start scheduler from app
def start_scheduler():
    scheduler.start()
    
# Function to stop scheduler
def stop_scheduler():
    scheduler.stop()