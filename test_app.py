#!/usr/bin/env python3
"""
Test script for the stock notification application
"""
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_ingestion.market_data import MarketDataFetcher
        print("✓ MarketDataFetcher imported successfully")
    except Exception as e:
        print(f"✗ MarketDataFetcher import failed: {e}")
        return False
    
    try:
        from data_ingestion.news_sentiment import NewsSentimentAnalyzer
        print("✓ NewsSentimentAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ NewsSentimentAnalyzer import failed: {e}")
        return False
    
    try:
        from analysis.market_analysis import MarketAnalyzer
        print("✓ MarketAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ MarketAnalyzer import failed: {e}")
        return False
    
    try:
        from analysis.recommendation import RecommendationEngine
        print("✓ RecommendationEngine imported successfully")
    except Exception as e:
        print(f"✗ RecommendationEngine import failed: {e}")
        return False
    
    try:
        from app import app
        print("✓ Flask app imported successfully")
    except Exception as e:
        print(f"✗ Flask app import failed: {e}")
        return False
    
    return True

def test_market_data():
    """Test market data fetching"""
    print("\nTesting market data fetching...")
    
    try:
        from data_ingestion.market_data import MarketDataFetcher
        
        fetcher = MarketDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Test with a simple stock
        data = fetcher.get_stock_data("AAPL", start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"✓ Market data fetched successfully for AAPL")
            print(f"  Data shape: {data.shape}")
            print(f"  Columns: {list(data.columns)}")
            return True
        else:
            print("✗ No data returned for AAPL")
            return False
            
    except Exception as e:
        print(f"✗ Market data test failed: {e}")
        return False

def test_analysis():
    """Test analysis functionality"""
    print("\nTesting analysis functionality...")
    
    try:
        from data_ingestion.market_data import MarketDataFetcher
        from analysis.market_analysis import MarketAnalyzer
        from analysis.recommendation import RecommendationEngine
        
        # Get some test data
        fetcher = MarketDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        data = fetcher.get_stock_data("AAPL", start_date, end_date)
        
        if data is None or data.empty:
            print("✗ No data available for analysis test")
            return False
        
        # Test market analysis
        analyzer = MarketAnalyzer()
        analysis_result = analyzer.analyze_stock(data, "AAPL")
        
        if analysis_result:
            print("✓ Market analysis completed successfully")
            print(f"  Technical indicators: {len(analysis_result.get('technical_indicators', {}))}")
            print(f"  Predictions: {len(analysis_result.get('predictions', {}))}")
            print(f"  Risk metrics: {len(analysis_result.get('risk_metrics', {}))}")
        else:
            print("✗ Market analysis failed")
            return False
        
        # Test recommendation engine
        recommendation_engine = RecommendationEngine()
        recommendation = recommendation_engine.generate_recommendation(
            data,
            analysis_result.get('technical_indicators', {}),
            0.1,  # dummy sentiment score
            analysis_result.get('regime', {})
        )
        
        if recommendation:
            print("✓ Recommendation engine completed successfully")
            print(f"  Recommendation: {recommendation.get('recommendation', 'N/A')}")
            print(f"  Confidence: {recommendation.get('confidence', 'N/A')}")
        else:
            print("✗ Recommendation engine failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app creation"""
    print("\nTesting Flask app...")
    
    try:
        from app import app
        
        # Test if app has the expected routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/api/stock/<symbol>']
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Route {route} found")
            else:
                print(f"✗ Route {route} not found")
                return False
        
        print("✓ Flask app created successfully with expected routes")
        return True
        
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Stock Notification Application - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Market Data Test", test_market_data),
        ("Analysis Test", test_analysis),
        ("Flask App Test", test_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo run the application:")
        print("  python app.py")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 