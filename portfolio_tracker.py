import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

class PortfolioTracker:
    def __init__(self, portfolio_file='portfolio.json'):
        self.portfolio_file = portfolio_file
        self.portfolio = self.load_portfolio()
        self.performance_history = []
        
    def load_portfolio(self):
        """Load portfolio from JSON file or return default if file doesn't exist"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default portfolio
                return {}
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return {}
            
    def save_portfolio(self):
        """Save portfolio to JSON file"""
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving portfolio: {e}")
            return False
            
    def add_position(self, symbol, shares, buy_price, buy_date=None):
        """Add a new position to the portfolio"""
        if buy_date is None:
            buy_date = datetime.now().strftime('%Y-%m-%d')
            
        if symbol in self.portfolio:
            # Update existing position with average price
            current_shares = self.portfolio[symbol]['shares']
            current_price = self.portfolio[symbol]['buy_price']
            total_shares = current_shares + shares
            total_cost = (current_shares * current_price) + (shares * buy_price)
            avg_price = total_cost / total_shares
            
            self.portfolio[symbol] = {
                'shares': total_shares,
                'buy_price': avg_price,
                'buy_date': buy_date
            }
        else:
            # Add new position
            self.portfolio[symbol] = {
                'shares': shares,
                'buy_price': buy_price,
                'buy_date': buy_date
            }
            
        return self.save_portfolio()
        
    def remove_position(self, symbol, shares=None):
        """Remove a position from the portfolio"""
        if symbol not in self.portfolio:
            return False
            
        if shares is None or shares >= self.portfolio[symbol]['shares']:
            # Remove entire position
            del self.portfolio[symbol]
        else:
            # Reduce position
            self.portfolio[symbol]['shares'] -= shares
            
        return self.save_portfolio()
        
    def get_portfolio_performance(self):
        """Calculate current portfolio performance"""
        performance = []
        total_value = 0
        total_cost = 0
        
        for symbol, details in self.portfolio.items():
            try:
                # Get current price
                stock = yf.Ticker(symbol)
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                
                shares = details['shares']
                buy_price = details['buy_price']
                
                current_value = current_price * shares
                cost_basis = buy_price * shares
                profit_loss = current_value - cost_basis
                profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
                
                total_value += current_value
                total_cost += cost_basis
                
                performance.append({
                    'symbol': symbol,
                    'shares': shares,
                    'buy_price': buy_price,
                    'current_price': current_price,
                    'current_value': round(current_value, 2),
                    'cost_basis': round(cost_basis, 2),
                    'profit_loss': round(profit_loss, 2),
                    'profit_loss_pct': round(profit_loss_pct, 2)
                })
            except Exception as e:
                print(f"Error calculating performance for {symbol}: {e}")
                continue
                
        # Sort by profit/loss percentage
        performance = sorted(performance, key=lambda x: x['profit_loss_pct'], reverse=True)
        
        return {
            'positions': performance,
            'total_value': round(total_value, 2),
            'total_cost': round(total_cost, 2),
            'total_profit_loss': round(total_value - total_cost, 2),
            'total_profit_loss_pct': round(((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0, 2),
            'timestamp': datetime.now().isoformat()
        }
        
    def track_performance_history(self, days=30):
        """Track portfolio performance over time"""
        today = datetime.now().date()
        start_date = today - timedelta(days=days)
        
        # Get historical prices for each position
        historical_data = {}
        for symbol in self.portfolio:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date, end=today + timedelta(days=1))
                historical_data[symbol] = hist
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
                continue
                
        # Calculate daily portfolio value
        daily_values = []
        date_range = pd.date_range(start=start_date, end=today)
        
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            daily_value = 0
            daily_cost = 0
            
            for symbol, details in self.portfolio.items():
                if symbol in historical_data:
                    # Get price on this date if available
                    hist = historical_data[symbol]
                    if date in hist.index:
                        price = hist.loc[date, 'Close']
                        shares = details['shares']
                        buy_price = details['buy_price']
                        
                        daily_value += price * shares
                        daily_cost += buy_price * shares
            
            if daily_cost > 0:
                daily_profit_pct = ((daily_value - daily_cost) / daily_cost) * 100
                daily_values.append({
                    'date': date_str,
                    'value': round(daily_value, 2),
                    'profit_loss_pct': round(daily_profit_pct, 2)
                })
                
        return daily_values
        
    def get_portfolio_allocation(self):
        """Get portfolio allocation by sector and asset"""
        allocation = {
            'by_symbol': {},
            'by_sector': {}
        }
        
        total_value = 0
        sectors = {}
        
        for symbol, details in self.portfolio.items():
            try:
                # Get current price and sector info
                stock = yf.Ticker(symbol)
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                sector = stock.info.get('sector', 'Unknown')
                
                value = current_price * details['shares']
                total_value += value
                
                # Add to sector allocation
                if sector in sectors:
                    sectors[sector] += value
                else:
                    sectors[sector] = value
                    
                # Add to symbol allocation
                allocation['by_symbol'][symbol] = value
            except Exception as e:
                print(f"Error getting allocation for {symbol}: {e}")
                continue
                
        # Convert to percentages
        if total_value > 0:
            for symbol in allocation['by_symbol']:
                allocation['by_symbol'][symbol] = round((allocation['by_symbol'][symbol] / total_value) * 100, 2)
                
            for sector in sectors:
                allocation['by_sector'][sector] = round((sectors[sector] / total_value) * 100, 2)
                
        return allocation