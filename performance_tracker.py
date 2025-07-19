import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

class PerformanceTracker:
    def __init__(self, history_file='prediction_history.json'):
        self.history_file = history_file
        self.prediction_history = self.load_history()
        
    def load_history(self):
        """Load prediction history from JSON file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            print(f"Error loading prediction history: {e}")
            return []
            
    def save_history(self):
        """Save prediction history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.prediction_history, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving prediction history: {e}")
            return False
            
    def add_prediction(self, symbol, prediction, confidence, sentiment_score, price, timestamp=None):
        """Add a new prediction to history"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        self.prediction_history.append({
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'price_at_prediction': price,
            'timestamp': timestamp,
            'validated': False,
            'actual_movement': None,
            'price_after': None,
            'is_correct': None
        })
        
        # Keep history manageable (last 1000 predictions)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
            
        return self.save_history()
        
    def validate_prediction(self, symbol, current_price, days_since=1):
        """Validate a previous prediction for a symbol"""
        validated_count = 0
        
        for prediction in self.prediction_history:
            if prediction['symbol'] == symbol and not prediction['validated']:
                # Check if prediction is old enough to validate
                pred_time = datetime.fromisoformat(prediction['timestamp'])
                if datetime.now() - pred_time >= timedelta(days=days_since):
                    # Calculate actual movement
                    price_at_prediction = prediction['price_at_prediction']
                    price_change_pct = ((current_price - price_at_prediction) / price_at_prediction) * 100
                    
                    if price_change_pct > 1:  # 1% increase
                        actual_movement = 'RISE'
                    elif price_change_pct < -1:  # 1% decrease
                        actual_movement = 'FALL'
                    else:
                        actual_movement = 'STABLE'
                        
                    # Check if prediction was correct
                    is_correct = prediction['prediction'] == actual_movement
                    
                    # Update prediction record
                    prediction['validated'] = True
                    prediction['actual_movement'] = actual_movement
                    prediction['price_after'] = current_price
                    prediction['price_change_pct'] = round(price_change_pct, 2)
                    prediction['is_correct'] = is_correct
                    prediction['validation_date'] = datetime.now().isoformat()
                    
                    validated_count += 1
                    
        if validated_count > 0:
            self.save_history()
            
        return validated_count
        
    def get_accuracy_metrics(self, days=30, min_predictions=10):
        """Calculate prediction accuracy metrics"""
        # Filter recent predictions that have been validated
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        recent_predictions = [p for p in self.prediction_history 
                             if p['validated'] and p['timestamp'] >= cutoff_date]
        
        if len(recent_predictions) < min_predictions:
            return {
                'overall_accuracy': None,
                'rise_accuracy': None,
                'fall_accuracy': None,
                'stable_accuracy': None,
                'prediction_count': len(recent_predictions),
                'min_predictions_required': min_predictions
            }
            
        # Calculate overall accuracy
        correct_predictions = [p for p in recent_predictions if p['is_correct']]
        overall_accuracy = (len(correct_predictions) / len(recent_predictions)) * 100
        
        # Calculate accuracy by prediction type
        rise_predictions = [p for p in recent_predictions if p['prediction'] == 'RISE']
        fall_predictions = [p for p in recent_predictions if p['prediction'] == 'FALL']
        stable_predictions = [p for p in recent_predictions if p['prediction'] == 'STABLE']
        
        rise_correct = len([p for p in rise_predictions if p['is_correct']])
        fall_correct = len([p for p in fall_predictions if p['is_correct']])
        stable_correct = len([p for p in stable_predictions if p['is_correct']])
        
        rise_accuracy = (rise_correct / len(rise_predictions)) * 100 if rise_predictions else 0
        fall_accuracy = (fall_correct / len(fall_predictions)) * 100 if fall_predictions else 0
        stable_accuracy = (stable_correct / len(stable_predictions)) * 100 if stable_predictions else 0
        
        # Calculate accuracy by confidence level
        high_conf_predictions = [p for p in recent_predictions if p['confidence'] >= 70]
        med_conf_predictions = [p for p in recent_predictions if 50 <= p['confidence'] < 70]
        low_conf_predictions = [p for p in recent_predictions if p['confidence'] < 50]
        
        high_conf_correct = len([p for p in high_conf_predictions if p['is_correct']])
        med_conf_correct = len([p for p in med_conf_predictions if p['is_correct']])
        low_conf_correct = len([p for p in low_conf_predictions if p['is_correct']])
        
        high_conf_accuracy = (high_conf_correct / len(high_conf_predictions)) * 100 if high_conf_predictions else 0
        med_conf_accuracy = (med_conf_correct / len(med_conf_predictions)) * 100 if med_conf_predictions else 0
        low_conf_accuracy = (low_conf_correct / len(low_conf_predictions)) * 100 if low_conf_predictions else 0
        
        return {
            'overall_accuracy': round(overall_accuracy, 2),
            'rise_accuracy': round(rise_accuracy, 2),
            'fall_accuracy': round(fall_accuracy, 2),
            'stable_accuracy': round(stable_accuracy, 2),
            'high_conf_accuracy': round(high_conf_accuracy, 2),
            'med_conf_accuracy': round(med_conf_accuracy, 2),
            'low_conf_accuracy': round(low_conf_accuracy, 2),
            'prediction_count': len(recent_predictions),
            'rise_count': len(rise_predictions),
            'fall_count': len(fall_predictions),
            'stable_count': len(stable_predictions)
        }
        
    def get_accuracy_by_symbol(self, days=30):
        """Get prediction accuracy broken down by symbol"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        recent_predictions = [p for p in self.prediction_history 
                             if p['validated'] and p['timestamp'] >= cutoff_date]
        
        symbol_accuracy = {}
        
        for prediction in recent_predictions:
            symbol = prediction['symbol']
            if symbol not in symbol_accuracy:
                symbol_accuracy[symbol] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0
                }
                
            symbol_accuracy[symbol]['total'] += 1
            if prediction['is_correct']:
                symbol_accuracy[symbol]['correct'] += 1
                
        # Calculate accuracy percentage for each symbol
        for symbol in symbol_accuracy:
            total = symbol_accuracy[symbol]['total']
            correct = symbol_accuracy[symbol]['correct']
            symbol_accuracy[symbol]['accuracy'] = round((correct / total) * 100, 2) if total > 0 else 0
            
        # Sort by accuracy (descending)
        sorted_symbols = sorted(symbol_accuracy.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        return {symbol: data for symbol, data in sorted_symbols}
        
    def generate_accuracy_chart(self):
        """Generate accuracy chart as base64 encoded image"""
        try:
            # Get accuracy metrics for different time periods
            metrics_7d = self.get_accuracy_metrics(days=7)
            metrics_30d = self.get_accuracy_metrics(days=30)
            metrics_90d = self.get_accuracy_metrics(days=90)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Define time periods and metrics
            periods = ['7 Days', '30 Days', '90 Days']
            accuracies = [
                metrics_7d.get('overall_accuracy', 0) or 0,
                metrics_30d.get('overall_accuracy', 0) or 0,
                metrics_90d.get('overall_accuracy', 0) or 0
            ]
            
            # Create bar chart
            plt.bar(periods, accuracies, color='skyblue')
            plt.axhline(y=50, color='r', linestyle='--', alpha=0.7)  # 50% reference line
            
            # Add labels and title
            plt.xlabel('Time Period')
            plt.ylabel('Accuracy (%)')
            plt.title('Prediction Accuracy Over Time')
            
            # Add value labels on top of bars
            for i, v in enumerate(accuracies):
                plt.text(i, v + 1, f"{v:.1f}%", ha='center')
                
            # Set y-axis range
            plt.ylim(0, 100)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Error generating accuracy chart: {e}")
            return None