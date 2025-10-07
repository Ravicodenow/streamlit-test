#!/bin/bash

# Credit Card Fraud Detection - Single File Streamlit App Launcher
# ================================================================

echo "🚀 Starting Credit Card Fraud Detection App..."
echo "📱 Single File Streamlit Application"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

# Install requirements if not already installed
echo "📦 Installing/checking dependencies..."
pip3 install -r requirements.txt

echo ""
echo "🌟 Starting Streamlit app..."
echo "🌐 Access the app at: http://localhost:8501"
echo "📋 Instructions:"
echo "   1. Train the model using the sidebar"
echo "   2. Choose analysis method (Manual/Sample/Bulk)"
echo "   3. Analyze credit card transactions for fraud"
echo ""
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Start the Streamlit app
streamlit run streamlit_fraud_detector.py --server.port 8501 --server.address localhost