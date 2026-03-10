# analysis/metrics.py
# Read output/state_log.jsonl and produce evaluation graphs:
#  - queue time series: adaptive (measured) vs fixed-timing baseline (simulated)
#  - arrival/departure rates over time
#  - average queue bar chart and percent reduction
#  - queue histogram / CDF
#
# Usage:
#   python analysis/metrics.py
#
import os, json, math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
STATE_LOG = "output/state_log.jsonl"   # source produced during your prototype run
OUT_DIR = "analysis/figures"
SUMMARY_CSV = "analysis/summary.csv"

# Baseline (fixed timing) params: tune for your PPT numbers
BASELINE_CYCLE = 60.0   # seconds (total cycle time)
BASELINE_GREEN = 30.0   # seconds of green for the approach under study (A)
SATURATION_FLOW = 1.0   # veh/s when green (approx typical small intersection)
DT = 1.0                # simulation timestep (s) for baseline (use 1s)

# smoothing for arrival / departure plotting (in seconds)
SMOOTH_WINDOW_S = 5

# ---------- Helpers ----------
def load_state_log(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run state_estimator.py first.")
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                rows.append(j)
            except Exception:
                continue
    if not rows:
        raise ValueError("No valid lines found in state log.")
    df = pd.DataFrame(rows)
    # parse timestamp if present; else create simple index
    if "timestamp" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["timestamp"])
        except Exception:
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["ts"] = pd.to_datetime(pd.Series(range(len(df))), unit="s")
    df = df.sort_values("ts").reset_index(drop=True)
    # ensure numeric
    for col in ["queue_length", "arrival_rate", "departure_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df

def simulate_fixed_baseline(arrival_rate_series, dt=1.0, cycle=60.0, green=30.0, sat_flow=1.0):
    """
    Simulate fixed-timing queue evolution for the approach using arrival_rate_series (veh/s).
    arrival_rate_series: array of arrival rates sampled at dt interval (veh/s).
    Returns baseline_queue array of same length (float).
    """
    n = len(arrival_rate_series)
    q = 0.0
    baseline_q = np.zeros(n, dtype=float)
    for i in range(n):
        t = i * dt
        # arrivals in this dt
        arrivals = arrival_rate_series[i] * dt
        q += arrivals
        # is green for approach A at this time?
        phase_in_cycle = (t % cycle)
        if phase_in_cycle < green:
            # allow departures at saturation flow
            depart_possible = sat_flow * dt
            departed = min(q, depart_possible)
            q -= departed
        # else red -> no departures for approach A
        baseline_q[i] = q
    return baseline_q

def smooth(x, window):
    if window <= 1:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

# ---------- Main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading state log...")
    df = load_state_log(STATE_LOG)
    # create a simple numeric time index (seconds since start)
    start_ts = df["ts"].iloc[0]
    df["t_s"] = (df["ts"] - start_ts).dt.total_seconds()
    # Resample to 1s if needed: many logs are already 1s spaced. We'll make a uniform grid.
    t_max = int(math.ceil(df["t_s"].iloc[-1]))
    times = np.arange(0, t_max+1, 1.0)
    # Interpolate queue_length and arrival_rate onto uniform 1s grid
    queue_interp = np.interp(times, df["t_s"].to_numpy(), df["queue_length"].to_numpy())
    arrival_interp = np.interp(times, df["t_s"].to_numpy(), df["arrival_rate"].to_numpy())
    departure_interp = np.interp(times, df["t_s"].to_numpy(), df["departure_rate"].to_numpy())

    # Smooth arrival/departure for plotting
    smooth_w = int(max(1, SMOOTH_WINDOW_S))
    arrival_s = smooth(arrival_interp, smooth_w)
    departure_s = smooth(departure_interp, smooth_w)

    # Baseline simulation using the observed arrival rates
    baseline_q = simulate_fixed_baseline(arrival_interp, dt=1.0, cycle=BASELINE_CYCLE, green=BASELINE_GREEN, sat_flow=SATURATION_FLOW)

    # Compute summary metrics
    avgq_adaptive = float(np.mean(queue_interp))
    avgq_baseline = float(np.mean(baseline_q))
    maxq_adaptive = float(np.max(queue_interp))
    maxq_baseline = float(np.max(baseline_q))
    reduction_pct = (avgq_baseline - avgq_adaptive) / avgq_baseline * 100.0 if avgq_baseline > 0 else 0.0

    # Save summary
    summary = {
        "avgq_adaptive": avgq_adaptive,
        "avgq_baseline": avgq_baseline,
        "maxq_adaptive": maxq_adaptive,
        "maxq_baseline": maxq_baseline,
        "reduction_pct": reduction_pct,
        "total_seconds": int(t_max)
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)
    print("Summary saved to", SUMMARY_CSV)
    print("Summary:", summary)

    # -------- Plot 1: Queue time-series (Adaptive vs Baseline) --------
    plt.figure(figsize=(10,4))
    plt.plot(times, queue_interp, label="Adaptive (measured)", linewidth=2)
    plt.plot(times, baseline_q, label=f"Fixed baseline (cycle={int(BASELINE_CYCLE)}s, green={int(BASELINE_GREEN)}s)", linewidth=2, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue length (vehicles)")
    plt.title("Queue Length Over Time: Adaptive vs Fixed Baseline")
    plt.legend()
    plt.grid(alpha=0.2)
    out1 = os.path.join(OUT_DIR, "queue_timeseries.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Saved", out1)

    # -------- Plot 2: Arrival & Departure rates --------
    plt.figure(figsize=(10,4))
    plt.plot(times, arrival_s, label="Arrival rate (smoothed, veh/s)", color="tab:blue")
    plt.plot(times, departure_s, label="Departure rate (smoothed, veh/s)", color="tab:orange")
    plt.xlabel("Time (s)")
    plt.ylabel("veh / s")
    plt.title("Arrival and Departure Rates")
    plt.legend()
    plt.grid(alpha=0.2)
    out2 = os.path.join(OUT_DIR, "arrival_departure_rates.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Saved", out2)

    # -------- Plot 3: Average queue bar chart & percent reduction --------
    plt.figure(figsize=(6,4))
    bars = [avgq_baseline, avgq_adaptive]
    labels = ["Fixed baseline", "Adaptive (measured)"]
    colors = ["#d62728", "#2ca02c"]
    plt.bar(labels, bars, color=colors)
    plt.ylabel("Average queue length (vehicles)")
    plt.title(f"Average Queue: Adaptive vs Baseline (reduction {reduction_pct:.1f}%)")
    out3 = os.path.join(OUT_DIR, "avg_queue_bar.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=150)
    plt.close()
    print("Saved", out3)

    # -------- Plot 4: Queue histogram / CDF (adaptive) --------
    plt.figure(figsize=(6,4))
    values = queue_interp
    plt.hist(values, bins=30, density=True, alpha=0.6, label="pdf")
    # CDF
    sorted_vals = np.sort(values)
    cdf = np.arange(len(sorted_vals)) / float(len(sorted_vals))
    plt.twinx()
    plt.plot(sorted_vals, cdf, color="black", linewidth=1, label="CDF")
    plt.ylabel("CDF")
    plt.title("Queue Distribution (Adaptive)")
    out4 = os.path.join(OUT_DIR, "queue_hist_cdf.png")
    plt.savefig(out4, dpi=150)
    plt.close()
    print("Saved", out4)

    print("All figures saved to", OUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()