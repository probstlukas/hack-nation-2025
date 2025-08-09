#!/bin/bash

# FinDocGPT Startup Script
# This script starts both the Flask backend and React frontend

echo "🚀 Starting FinDocGPT - AI for Financial Document Analysis"
echo "Sponsored by AkashX.ai | HackNation 2025 Challenge"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Activate conda environment
echo "📦 Activating conda environment 'hack-nation'..."
eval "$(conda shell.bash hook)"
conda activate hack-nation

# Install Python dependencies if needed
echo "📦 Installing/updating Python dependencies..."
pip install -r requirements.txt

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "🔧 Starting services..."

# Start FastAPI backend in background
echo "🚀 Starting FastAPI backend on http://localhost:5001"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 5001 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start React frontend
echo "⚛️  Starting React frontend on http://localhost:3000"
cd frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Both services are starting up!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop both services"

# Handle Ctrl+C to stop both services
trap 'echo ""; echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

# Wait for either process to finish
wait
