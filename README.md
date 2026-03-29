# FlowSense

FlowSense is an intelligent traffic management system that combines video-based traffic sensing, signal control, and corridor-level simulation.

The project has three core parts:

- Traffic state extraction from video
- Signal control logic (fixed, adaptive, and emergency-priority behavior)
- Multi-intersection simulation and analysis

## Overview

FlowSense processes traffic video to estimate live traffic state, then uses that state to evaluate and compare different control strategies across intersections.

The system includes:

- Vehicle detection/tracking pipeline
- Queue and flow estimation
- Control logic for signal phase decisions
- Emergency corridor behavior for priority movement
- Dashboard and analysis outputs for comparison and monitoring

## Project Structure

```text
.
├── src/
│   ├── dashboard.py
│   ├── state_estimator.py
│   ├── detect_vehicles.py
│   ├── detect_track.py
│   ├── controller.py
│   ├── demo_runner.py
│   └── zones.py
├── output/
│   ├── state_log.jsonl
│   └── signal_log.jsonl
├── results/
├── analysis/
│   ├── metrics.py
│   ├── counterfactual.py
│   └── README.md
├── data/
├── run_demo.sh
├── requirements.txt
└── yolov8n.pt
```

## Core Modules

### `src/state_estimator.py`

- Reads traffic video
- Tracks vehicles using YOLO
- Estimates queue length, arrival rate, and departure rate
- Writes traffic state to `output/state_log.jsonl`

### `src/controller.py`

- Consumes latest traffic state
- Runs adaptive signal switching logic
- Logs decisions and signal state to `output/signal_log.jsonl`

### `src/dashboard.py`

- Runs live multi-intersection simulation
- Supports fixed, adaptive, and emergency modes
- Visualizes vehicles, signals, queues, and performance trends
- Includes video-derived comparison scenarios

### `src/demo_runner.py`

- Optional offline scenario runner
- Produces JSON result files under `results/`

### `analysis/`

- Generates evaluation charts and summary CSVs from recorded state logs

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If the above fails, install the minimum dependencies for run instead:

```bash
pip install --upgrade pip setuptools wheel
pip install streamlit plotly numpy pandas
python -c "import streamlit, plotly, numpy, pandas; print('OK')"
```

## How to Run

### 1) Generate traffic state from video

```bash
python src/state_estimator.py
```

This creates/updates:

- `output/state_log.jsonl`

### 2) Run the dashboard

```bash
./run_demo.sh
```

or:

```bash
streamlit run src/dashboard.py
```

### 3) Optional utilities

Offline scenario generation:

```bash
python src/demo_runner.py
```

Analysis scripts:

```bash
python analysis/metrics.py
python analysis/counterfactual.py
```

## Data and Outputs

### Input

- Traffic video files under `data/videos/`

### Runtime outputs

- `output/state_log.jsonl`: estimated traffic state over time
- `output/signal_log.jsonl`: signal/controller events and status

### Analysis outputs

- `analysis/summary.csv`
- `analysis/cf_summary.csv`
- Plots under `analysis/figures/` and `analysis/figures_cf/`

## Notes

- `yolov8n.pt` is used by detection/tracking scripts.
- If `output/state_log.jsonl` is missing, run the estimator before running state-driven comparisons.
- `run_demo.sh` starts the dashboard directly.
