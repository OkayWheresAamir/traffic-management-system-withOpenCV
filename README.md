# FlowSense: Intelligent Traffic Management Prototype

FlowSense is a traffic control prototype that combines:

- Computer vision based traffic state extraction from real video
- A live, step-based, stochastic multi-intersection simulation
- Adaptive and emergency-priority signal control
- Dashboard-driven comparison against fixed timing

The project is designed for rapid prototyping, demos, and judging/pitch presentations.

## 1) What This Project Demonstrates

- Live simulation (not replay): the dashboard updates one simulation tick per rerun.
- Stochastic behavior: vehicle arrivals and movement vary across runs.
- Multi-intersection corridor control: fixed vs adaptive timing.
- Emergency handling: ambulance dispatch with signal pre-clear (green-wave behavior).
- Video-derived benchmarking: traffic demand inferred from `state_log` is used to drive a wide-road corridor comparison at the bottom of the dashboard.

## 2) Architecture Overview

### Layer A: CV State Estimation
Input traffic video is processed with YOLO tracking to estimate:

- `queue_length`
- `arrival_rate` (veh/s over rolling window)
- `departure_rate` (veh/s over rolling window)

These values are logged to `output/state_log.jsonl`.

### Layer B: Local + Corridor Signal Logic
Signals are controlled by mode:

- `fixed`: cycle-based timing
- `adaptive`: pressure-based switching with min/max green, hysteresis, and platoon hold
- `emergency`: adaptive control with ambulance priority pre-clear

### Layer C: Visualization and Evaluation
Streamlit dashboard provides:

- Live animation + KPI updates from current run only
- Fixed vs adaptive comparison for completed live runs
- CV snapshot with video
- Video-derived, wide-road, multi-intersection benchmark (fixed vs adaptive)

## 3) Core Control Logic (Formulas)

### Pressure Metric
For each approach/group:

`Pressure = Queue + w * ArrivalRate`

where `w` is a tunable weight (arrival look-ahead).

### Adaptive Switching Rule
Switch green when:

- `time_in_phase >= MIN_GREEN`
- opposing pressure exceeds current pressure by `HYSTERESIS`

Force switch when:

- `time_in_phase >= MAX_GREEN`

### Delay Metric (Dashboard)

`avg_delay = cumulative_wait_time / observed_vehicles`

### Throughput

`throughput = cumulative vehicles completed/served`

### Emergency Priority (PCS-style behavior)

- Ambulance dispatch occurs early in run
- Junctions ahead are prioritized for EW green while ambulance approaches
- Non-emergency vehicles yield and avoid centerline blocking

## 4) Repository Structure

```text
.
├── src/
│   ├── dashboard.py          # Main Streamlit app (live simulation + benchmark views)
│   ├── state_estimator.py    # CV state extraction to output/state_log.jsonl
│   ├── detect_vehicles.py    # Simple YOLO vehicle detection
│   ├── detect_track.py       # Homography + bird-eye visualization
│   ├── demo_runner.py        # Optional offline scenario generator to results/*.json
│   ├── controller.py         # Earlier standalone adaptive controller prototype
│   └── zones.py              # Zone geometry helper
├── output/
│   ├── state_log.jsonl
│   └── signal_log.jsonl
├── results/                  # Offline JSON outputs (optional/legacy)
├── analysis/                 # Plotting + summary scripts
├── run_demo.sh               # Launches dashboard directly
├── requirements.txt
└── yolov8n.pt
```

## 5) Setup

### Prerequisites

- Python 3.10+ recommended
- macOS/Linux shell commands below (Windows also works with equivalent commands)
- `yolov8n.pt` available at repo root (already present in this project)

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 6) Run Modes

### A. Run the Live Dashboard (Primary Path)

```bash
./run_demo.sh
```

or

```bash
streamlit run src/dashboard.py
```

Open the local Streamlit URL shown in terminal.

### B. Generate CV Traffic State from Video

```bash
python src/state_estimator.py
```

This writes `output/state_log.jsonl` (used by the dashboard CV and benchmark sections).

### C. Optional: Run Detection / Homography Views

```bash
python src/detect_vehicles.py --video data/videos/traffic.mp4 --output output/annotated.mp4
python src/detect_track.py
```

### D. Optional Offline Scenario Generator

```bash
python src/demo_runner.py
```

Writes scenario JSON files to `results/`.
The current live dashboard does not depend on these files for top simulation.

### E. Analysis Plots

```bash
python analysis/metrics.py
python analysis/counterfactual.py
```

Outputs are saved under `analysis/figures/` and `analysis/figures_cf/`.

## 7) Dashboard Walkthrough

### Top Section: Live Engine

- Mode buttons: `Fixed Timing`, `FlowSense Adaptive`, `Emergency Corridor`
- One-step-per-rerun simulation
- KPIs updated from current run state only:
  - Sim Time
  - Queue Length
  - Throughput
  - Avg Delay
  - Avg Stops
  - Active Vehicles

### Mid Section

- Fixed vs adaptive charts from completed live runs
- CV snapshot and currently analyzed traffic video

### Bottom Section

- Video-derived wide-road corridor benchmark
- Multi-intersection fixed vs adaptive comparison
- Snapshot slider for side-by-side corridor state view

## 8) Important Output Files

- `output/state_log.jsonl`: CV-derived traffic state stream
- `output/signal_log.jsonl`: controller logs (when using controller script)
- `analysis/summary.csv`: baseline/adaptive summary
- `analysis/cf_summary.csv`: counterfactual summary
- `analysis/figures/*.png`: analysis figures

## 9) Tuning Tips

You can tune behavior by editing constants in:

- `src/dashboard.py` for live simulation and video-derived benchmark parameters
- `src/state_estimator.py` for detection thresholds and rolling window
- `src/demo_runner.py` for offline scenario assumptions

Recommended tuning targets:

- Signal responsiveness: `MIN_GREEN`, `MAX_GREEN`, `HYSTERESIS`
- Congestion stability: spawn rates and platoon hold thresholds
- Emergency behavior: dispatch time and look-ahead distance

## 10) Troubleshooting

- Dashboard opens but no CV data:
  - Run `python src/state_estimator.py` first to create/update `output/state_log.jsonl`.
- CV scripts are slow:
  - Lower input resolution, use smaller model, or run on GPU-enabled environment.
- Streamlit import/runtime warnings outside Streamlit:
  - Expected when importing `dashboard.py` directly in bare Python.
- Missing video in CV snapshot:
  - Dashboard checks `output/detection.mp4`, then `data/videos/traffic.mp4`.

## 11) Current Status

- Live simulation is session-state driven and stochastic.
- Emergency dispatch and signal pre-clear are visible.
- Vehicle overlap handling and platoon flow are improved.
- Video-derived bottom benchmark now supports a wide-road multi-intersection corridor comparison.
