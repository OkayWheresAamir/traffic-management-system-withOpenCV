# analysis/counterfactual.py
# Run a counterfactual adaptive-vs-fixed simulation using logged arrival rates (from state_log.jsonl).
# Produces queue timeseries and comparison figures in analysis/figures_cf/
import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
STATE_LOG = "output/state_log.jsonl"
OUT_DIR = "analysis/figures_cf"
WINDOW = 10.0  # smoothing window (s) for plots

# Baseline fixed timing
BASELINE_CYCLE = 60.0
BASELINE_GREEN = 30.0
SAT_FLOW = 1.0  # veh/s departure during green

# Adaptive controller params (for simulation)
MIN_GREEN = 8.0
MAX_GREEN = 45.0
YELLOW = 3.0
ALL_RED = 1.0
ARRIVAL_WEIGHT = 5.0
HYST = 1.0
EVAL_DT = 1.0  # simulation step in seconds

# Simulated opposing approach B (to create contention)
LAMBDA_B_PER_MIN = 20.0
SAT_B = 1.0

# ---------- Helpers ----------
def load_state_log(path):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        raise ValueError("No data in state log.")
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["ts"] = pd.to_datetime(pd.Series(range(len(df))), unit="s")
    df = df.sort_values("ts").reset_index(drop=True)
    for col in ["queue_length", "arrival_rate", "departure_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df

def synthesize_per_second_arrivals(arrival_rate_series, dt=1.0):
    # arrival_rate_series is veh/s sampled at dt; convert to integer arrivals per second
    # We'll round probabilistically: arrivals_per_second = Poisson(arrival_rate * dt)
    arrivals = [np.random.poisson(max(0.0, r*dt)) for r in arrival_rate_series]
    return np.array(arrivals, dtype=int)

# Baseline simulation: same as earlier
def simulate_fixed(arrivals, dt=1.0, cycle=60.0, green=30.0, sat_flow=1.0):
    n = len(arrivals)
    q = 0.0
    q_ts = np.zeros(n)
    for i in range(n):
        q += arrivals[i]  # arrivals in this second
        t = i * dt
        if (t % cycle) < green:
            # departures limited by saturation
            depart = min(q, sat_flow * dt)
            q -= depart
        q_ts[i] = q
    return q_ts

# Adaptive simulation: pressure-based control on two approaches A and B
def simulate_adaptive(arrivals_A, lambda_B_per_min, sim_dt=1.0):
    n = len(arrivals_A)
    qA = 0.0
    qB = 0.0
    qA_ts = np.zeros(n)
    qB_ts = np.zeros(n)
    # simulate B arrivals by Poisson with mean lambda_B_per_sec
    lambda_B_per_sec = lambda_B_per_min / 60.0
    # initial phase
    phase = "A_GREEN"
    phase_time = 0.0
    t = 0.0
    i = 0
    # we'll step per-second
    while i < n:
        # arrivals this second
        aA = arrivals_A[i]
        aB = np.random.poisson(lambda_B_per_sec * sim_dt)
        qA += aA
        qB += aB
        # compute approximate arrival rates over recent WINDOW for pressure (use simple moving avg)
        # For speed, use a small approximate: arrival_rate_est_A = aA (per-second)
        arr_rate_A = aA / sim_dt
        arr_rate_B = aB / sim_dt
        # compute pressures
        pA = qA + ARRIVAL_WEIGHT * arr_rate_A
        pB = qB + ARRIVAL_WEIGHT * arr_rate_B
        # decision (respect min green)
        if phase_time < MIN_GREEN:
            decision = "HOLD"
        else:
            # check hysteresis
            if phase == "A_GREEN" and pB > pA + HYST:
                decision = "SWITCH"
            elif phase == "B_GREEN" and pA > pB + HYST:
                decision = "SWITCH"
            elif phase_time >= MAX_GREEN:
                decision = "SWITCH"
            else:
                decision = "HOLD"
        # execute departures depending on phase
        if phase == "A_GREEN":
            # A discharges at sat_flow
            departA = min(qA, SAT_FLOW * sim_dt)
            qA -= departA
            # B does not discharge
        else:
            departB = min(qB, SAT_B * sim_dt)
            qB -= departB
        # apply switch if decided
        if decision == "SWITCH":
            # simulate yellow + all-red delay (we'll consume those seconds with no departures)
            # For simplicity, model them as instantaneous but advance phase_time accordingly
            # To keep alignment with per-second steps, we just switch phase and reset phase_time
            phase = "B_GREEN" if phase == "A_GREEN" else "A_GREEN"
            phase_time = 0.0
        else:
            phase_time += sim_dt
        # record
        qA_ts[i] = qA
        qB_ts[i] = qB
        # advance
        i += 1
        t += sim_dt
    return qA_ts, qB_ts

# ---------- Main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_state_log(STATE_LOG)
    # uniform time grid
    start_ts = df["ts"].iloc[0]
    times = ((df["ts"] - start_ts).dt.total_seconds()).to_numpy()
    t_max = int(math.ceil(times[-1]))
    grid = np.arange(0, t_max+1)
    # Interpolate arrival_rate to grid (veh/s)
    arrival_interp = np.interp(grid, times, df["arrival_rate"].to_numpy())
    # synthesize integer arrivals per second (stochastic)
    arrivals_A = synthesize_per_second_arrivals(arrival_interp, dt=1.0)
    # run baseline
    baseline_q = simulate_fixed(arrivals_A, dt=1.0, cycle=BASELINE_CYCLE, green=BASELINE_GREEN, sat_flow=SAT_FLOW)
    # run adaptive (counterfactual)
    np.random.seed(0)  # reproducible
    adapt_qA, adapt_qB = simulate_adaptive(arrivals_A, lambda_B_per_min=LAMBDA_B_PER_MIN, sim_dt=1.0)

    # metrics
    avg_baseline = baseline_q.mean()
    avg_adapt = adapt_qA.mean()
    max_baseline = baseline_q.max()
    max_adapt = adapt_qA.max()
    reduction = (avg_baseline - avg_adapt) / avg_baseline * 100 if avg_baseline>0 else 0.0

    summary = {"avg_baseline": avg_baseline, "avg_adapt": avg_adapt,
               "max_baseline": float(max_baseline), "max_adapt": float(max_adapt),
               "reduction_pct": reduction, "seconds": len(grid)}
    pd = __import__("pandas")
    pd.DataFrame([summary]).to_csv("analysis/cf_summary.csv", index=False)
    print("Summary:", summary)

    # plots: queue timeseries
    plt.figure(figsize=(10,4))
    plt.plot(grid, baseline_q, label="Fixed baseline", alpha=0.9)
    plt.plot(grid, adapt_qA, label="Adaptive (simulated)", alpha=0.9)
    plt.xlabel("Time (s)"); plt.ylabel("Queue (veh)")
    plt.title("Counterfactual: Fixed vs Adaptive (using measured arrivals)")
    plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "cf_queue_timeseries.png"), dpi=150)
    plt.close()

    # bar chart
    plt.figure(figsize=(5,4))
    plt.bar(["Fixed","Adaptive"], [avg_baseline, avg_adapt], color=["#d62728","#2ca02c"])
    plt.title(f"Avg queue (reduction {reduction:.1f}%)")
    plt.ylabel("vehicles")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "cf_avg_bar.png"), dpi=150)
    plt.close()

    print("Counterfactual figures saved to", OUT_DIR)

if __name__ == "__main__":
    main()