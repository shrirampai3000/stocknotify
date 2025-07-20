#!/usr/bin/env python3
"""
Startup script for the Stock Notification Application
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'yfinance', 'requests', 
        'textblob', 'beautifulsoup4', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        install = input("Would you like to install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            print("Installing missing packages...")
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Installation complete!")
        else:
            print("⚠️  Some features may not work without these packages.")
    
    return len(missing_packages) == 0

def check_environment():
    """Check environment setup"""
    print("\n🔧 Checking environment...")
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("  ℹ️  No .env file found (optional)")
        print("  ℹ️  Create .env file for API keys if needed")
    else:
        print("  ✓ .env file found")
    
    # Check if templates and static directories exist
    if Path('templates').exists():
        print("  ✓ Templates directory found")
    else:
        print("  ✗ Templates directory missing")
        return False
    
    if Path('static').exists():
        print("  ✓ Static directory found")
    else:
        print("  ✗ Static directory missing")
        return False
    
    return True

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    
    try:
        result = subprocess.run([sys.executable, 'test_app.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("⚠️  Test file not found, skipping tests")
        return True

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting Stock Notification Application...")
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    try:
        # Import and run the app
        from app import app
        
        print("✅ Application loaded successfully!")
        print("\n🌐 Opening browser...")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("📊 Dashboard will open automatically in your browser")
        print("🔗 Manual link: http://localhost:5000")
        print("\n⏹️  Press Ctrl+C to stop the application")
        print("=" * 50)
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except ImportError as e:
        print(f"❌ Failed to import application: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False

def main():
    """Main function"""
    print("📈 Stock Notification Application")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ app.py not found in current directory")
        print("💡 Make sure you're in the project root directory")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("⚠️  Some dependencies are missing")
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return False
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but continuing...")
    
    # Start application
    return start_application()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 