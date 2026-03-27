#!/bin/bash
# FlowSense — Hackathon Demo Launcher
# Run this script to generate results from real video data and launch the dashboard.

set -e

echo "=========================================="
echo "  FlowSense Demo"
echo "  AI-Driven Adaptive Traffic & Emergency"
echo "  Corridor System"
echo "=========================================="
echo ""

echo "[1/2] Running simulations on video-derived traffic state..."
python src/demo_runner.py
echo ""

echo "[2/2] Launching dashboard..."
echo "       (Press Ctrl+C to stop)"
echo ""
streamlit run src/dashboard.py --server.headless true
