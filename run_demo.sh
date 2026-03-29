#!/bin/bash
# FlowSense — Live Dashboard Launcher
# Launches the session-state live simulation directly.

set -e

echo "=========================================="
echo "  FlowSense Live Simulation"
echo "  Adaptive Traffic & Emergency Corridor"
echo "=========================================="
echo ""
echo "Launching Streamlit dashboard..."
echo "(Press Ctrl+C to stop)"
echo ""

streamlit run src/dashboard.py --server.headless true
