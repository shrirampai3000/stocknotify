import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
import json

class NotificationSystem:
    def __init__(self, config_file='notification_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.notification_history = []
        
    def load_config(self):
        """Load notification configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default config
                return {
                    'email': {
                        'enabled': False,
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'username': '',
                        'password': '',
                        'recipients': []
                    },
                    'thresholds': {
                        'price_change': 5.0,  # 5% price change
                        'sentiment_change': 0.3,  # 0.3 sentiment score change
                        'volume_spike': 200  # 200% volume increase
                    },
                    'notification_types': {
                        'price_alerts': True,
                        'sentiment_alerts': True,
                        'portfolio_alerts': True,
                        'daily_summary': True
                    }
                }
        except Exception as e:
            print(f"Error loading notification config: {e}")
            return {}
            
    def save_config(self):
        """Save notification configuration to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving notification config: {e}")
            return False
            
    def send_email_notification(self, subject, message, recipients=None):
        """Send email notification"""
        if not self.config.get('email', {}).get('enabled', False):
            return False
            
        try:
            email_config = self.config['email']
            
            if not recipients:
                recipients = email_config.get('recipients', [])
                
            if not recipients:
                return False
                
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            # Log notification
            self.notification_history.append({
                'type': 'email',
                'subject': subject,
                'recipients': recipients,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            print(f"Error sending email notification: {e}")
            return False
            
    def generate_stock_alert(self, stock_data, alert_type):
        """Generate stock alert notification"""
        if not self.config.get('notification_types', {}).get(f'{alert_type}_alerts', False):
            return False
            
        try:
            symbol = stock_data['symbol']
            current_price = stock_data['current_price']
            change_pct = stock_data.get('change_pct', 0)
            recommendation = stock_data.get('recommendation', '')
            
            subject = f"Stock Alert: {symbol} - {recommendation}"
            
            # Create HTML message
            message = f"""
            <html>
            <body>
                <h2>Stock Alert: {symbol}</h2>
                <p><strong>Current Price:</strong> {current_price}</p>
                <p><strong>Change (%):</strong> {change_pct}%</p>
                <p><strong>Recommendation:</strong> {recommendation}</p>
                <p><strong>Alert Type:</strong> {alert_type}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """
            
            return self.send_email_notification(subject, message)
        except Exception as e:
            print(f"Error generating stock alert: {e}")
            return False
            
    def generate_portfolio_summary(self, portfolio_data):
        """Generate portfolio summary notification"""
        if not self.config.get('notification_types', {}).get('daily_summary', False):
            return False
            
        try:
            total_value = portfolio_data['total_value']
            total_profit_loss = portfolio_data['total_profit_loss']
            total_profit_loss_pct = portfolio_data['total_profit_loss_pct']
            positions = portfolio_data['positions']
            
            subject = f"Portfolio Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create HTML message
            message = f"""
            <html>
            <body>
                <h2>Portfolio Summary</h2>
                <p><strong>Total Value:</strong> ${total_value}</p>
                <p><strong>Total Profit/Loss:</strong> ${total_profit_loss} ({total_profit_loss_pct}%)</p>
                
                <h3>Positions:</h3>
                <table border="1" cellpadding="5">
                    <tr>
                        <th>Symbol</th>
                        <th>Shares</th>
                        <th>Current Price</th>
                        <th>Current Value</th>
                        <th>Profit/Loss</th>
                        <th>Profit/Loss (%)</th>
                    </tr>
            """
            
            # Add positions to table
            for position in positions:
                profit_loss_class = 'color:green;' if position['profit_loss'] >= 0 else 'color:red;'
                message += f"""
                    <tr>
                        <td>{position['symbol']}</td>
                        <td>{position['shares']}</td>
                        <td>${position['current_price']}</td>
                        <td>${position['current_value']}</td>
                        <td style="{profit_loss_class}">${position['profit_loss']}</td>
                        <td style="{profit_loss_class}">{position['profit_loss_pct']}%</td>
                    </tr>
                """
                
            message += """
                </table>
            </body>
            </html>
            """
            
            return self.send_email_notification(subject, message)
        except Exception as e:
            print(f"Error generating portfolio summary: {e}")
            return False
            
    def check_and_send_alerts(self, stock_data_list, portfolio=None):
        """Check for alert conditions and send notifications"""
        alerts_sent = 0
        
        for stock_data in stock_data_list:
            try:
                symbol = stock_data['symbol']
                change_pct = stock_data.get('change_pct', 0)
                sentiment_score = stock_data.get('sentiment_score', 0)
                
                # Check if stock is in portfolio
                in_portfolio = portfolio and symbol in portfolio
                
                # Price change alert
                if abs(change_pct) >= self.config['thresholds']['price_change']:
                    if self.generate_stock_alert(stock_data, 'price'):
                        alerts_sent += 1
                        
                # Sentiment change alert
                if abs(sentiment_score) >= self.config['thresholds']['sentiment_change']:
                    if self.generate_stock_alert(stock_data, 'sentiment'):
                        alerts_sent += 1
                        
                # Portfolio specific alerts (higher priority)
                if in_portfolio and abs(change_pct) >= (self.config['thresholds']['price_change'] / 2):
                    if self.generate_stock_alert(stock_data, 'portfolio'):
                        alerts_sent += 1
            except Exception as e:
                print(f"Error checking alerts for {stock_data.get('symbol', 'unknown')}: {e}")
                continue
                
        return alerts_sent