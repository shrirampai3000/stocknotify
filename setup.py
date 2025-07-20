"""
Setup script for Stock Notification Application
"""
from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-notification-app",
    version="1.0.0",
    author="Stock Notification Team",
    author_email="your-email@example.com",
    description="A comprehensive stock market analysis and notification system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-notification-app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "full": [
            "prophet>=1.1.4",
            "arch>=6.2.0",
            "ruptures>=1.1.7",
            "lightgbm>=4.1.0",
            "xgboost>=2.0.3",
            "shap>=0.44.0",
            "alpha-vantage>=2.3.1",
            "fredapi>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-notify=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*", "static/*"],
    },
    keywords="stock market analysis trading finance investment",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stock-notification-app/issues",
        "Source": "https://github.com/yourusername/stock-notification-app",
        "Documentation": "https://github.com/yourusername/stock-notification-app#readme",
    },
) 