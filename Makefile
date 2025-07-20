# Stock Notification Application - Makefile

.PHONY: help install test run clean lint format

# Default target
help:
	@echo "Stock Notification Application - Available Commands:"
	@echo ""
	@echo "  install    - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  run        - Start the application"
	@echo "  clean      - Clean up cache and temporary files"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black"
	@echo "  setup      - Initial setup (install + test)"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Run tests
test:
	@echo "Running tests..."
	python test_app.py

# Start the application
run:
	@echo "Starting Stock Notification Application..."
	python app.py

# Start with startup script
start:
	@echo "Starting with startup script..."
	python start_app.py

# Clean up cache and temporary files
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*.pyd" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Run code linting
lint:
	@echo "Running code linting..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code with black
format:
	@echo "Formatting code with black..."
	black . --line-length=127

# Initial setup
setup: install test
	@echo "Setup complete!"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/macOS"
	@echo "  venv\\Scripts\\activate     # On Windows"

# Install in development mode
install-dev-mode:
	@echo "Installing in development mode..."
	pip install -e .

# Build package
build:
	@echo "Building package..."
	python setup.py sdist bdist_wheel

# Install full version with all optional dependencies
install-full:
	@echo "Installing full version with all dependencies..."
	pip install -r requirements.txt
	pip install prophet arch ruptures lightgbm xgboost shap alpha-vantage fredapi

# Health check
health:
	@echo "Running health check..."
	curl -f http://localhost:5000/health || echo "Application not running"

# Docker commands (if using Docker)
docker-build:
	@echo "Building Docker image..."
	docker build -t stock-notification-app .

docker-run:
	@echo "Running Docker container..."
	docker run -p 5000:5000 stock-notification-app

# Git helpers
git-init:
	@echo "Initializing git repository..."
	git init
	git add .
	git commit -m "Initial commit"

git-push:
	@echo "Pushing to remote repository..."
	git add .
	git commit -m "Update: $(shell date)"
	git push 