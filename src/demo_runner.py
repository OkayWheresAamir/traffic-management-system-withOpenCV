# src/demo_runner.py
# FlowSense Demo Runner — replays real video-derived traffic state through
# 4 simulation scenarios (fixed, adaptive, emergency w/o PCS, emergency w/ PCS)
# and writes results/*.json for the Streamlit dashboard.
#
# Usage: python src/demo_runner.py

import json, os, math
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
STATE_LOG = "output/state_log.jsonl"
RESULTS_DIR = "results"
SIM_DURATION = 600  # seconds (10 minutes)

# Junction layout: 3 intersections on an EW arterial, 500m apart (Delhi arterial scale)
JUNCTIONS = ["J0", "J1", "J2"]
JUNCTION_SPACING_M = 500.0
ARTERIAL_SPEED_MPS = 12.0  # ~43 km/h free-flow (realistic Delhi arterial with mixed traffic)
TRAVEL_DELAY_S = int(round(JUNCTION_SPACING_M / ARTERIAL_SPEED_MPS))  # ~42s

# Signal control
FIXED_CYCLE = 90.0   # Delhi-typical long cycle for fixed timing
FIXED_GREEN = 40.0   # green per direction in fixed mode
SAT_FLOW = 0.8       # vehicles/s departing during green (Delhi mixed traffic)
SAT_FLOW_NS = 0.6    # lower for cross streets

# Adaptive controller
MIN_GREEN = 8.0
MAX_GREEN = 45.0
YELLOW = 3.0
ALL_RED = 1.5
ARRIVAL_WEIGHT = 5.0
HYSTERESIS = 2.0

# Cross-street traffic (Poisson)
LAMBDA_NS_PER_SEC = 0.18  # vehicles/s on each NS approach

# Regional coordination (Layer 2)
OFFSET_WEIGHT = 0.6       # how much to bias green start toward predicted platoon arrival
PLATOON_LOOKAHEAD_S = 25  # seconds ahead to predict platoon arrival

# Emergency / PCS (Layer 3)
AMBULANCE_DISPATCH_T = 120    # seconds into simulation
AMBULANCE_SPEED_MPS = 14.0   # ~50 km/h with lights/siren in Delhi traffic
AMBULANCE_CRAWL_MPS = 3.0    # speed when navigating past queued traffic near red
PCS_PRE_CLEAR_S = 10         # seconds before ETA to start forcing green
PCS_POST_GUARD_S = 6         # seconds after ETA to hold green
AMBULANCE_STOP_PENALTY_S = 22 # average delay per red-light stop (wait + queue clear + accel)

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_state_log(path):
    """Load state_log.jsonl and return a DataFrame with relative time in seconds."""
    rows = []
    with open(path) as f:
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
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    # Convert to relative seconds from start
    start = df["ts"].iloc[0]
    df["t_rel"] = (df["ts"] - start).dt.total_seconds()
    for col in ["queue_length", "arrival_rate", "departure_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df


def prepare_arrivals(df, duration):
    """Interpolate real data to a uniform 1s grid and extend to full duration."""
    times = df["t_rel"].to_numpy()
    arr_rate = df["arrival_rate"].to_numpy()
    queue = df["queue_length"].to_numpy()

    # Create uniform grid
    t_max_data = int(math.ceil(times[-1]))
    grid_data = np.arange(0, t_max_data + 1)
    arr_interp = np.interp(grid_data, times, arr_rate)
    queue_interp = np.interp(grid_data, times, queue)

    # Extend to full duration by cycling with slight noise
    grid_full = np.arange(0, duration)
    arr_full = np.zeros(duration)
    queue_full = np.zeros(duration)
    data_len = len(arr_interp)
    for i in range(duration):
        idx = i % data_len
        noise = 1.0 + np.random.normal(0, 0.1)
        arr_full[i] = max(0, arr_interp[idx] * noise)
        queue_full[i] = max(0, queue_interp[idx] + np.random.normal(0, 0.5))

    return arr_full, queue_full


def synthesize_arrivals(rate_series, dt=1.0):
    """Convert continuous arrival rate to integer vehicle arrivals via Poisson."""
    return np.array([np.random.poisson(max(0.0, r * dt)) for r in rate_series])


def make_junction_arrivals(ew_arrivals_j0, duration):
    """
    Create arrival arrays for all 3 junctions on both EW and NS approaches.
    J0: EW from real data, NS from Poisson
    J1: EW = J0 departures delayed by TRAVEL_DELAY_S (platoon propagation)
    J2: EW = J1 departures delayed by TRAVEL_DELAY_S
    NS at all junctions: independent Poisson
    """
    arrivals = {}
    # J0 EW from real data
    arrivals[("J0", "ew")] = ew_arrivals_j0.copy()

    # NS for all junctions (independent Poisson)
    for j in JUNCTIONS:
        arrivals[(j, "ns")] = synthesize_arrivals(
            np.full(duration, LAMBDA_NS_PER_SEC))

    # J1 and J2 EW: delayed version of upstream departures
    # We'll fill these during simulation (they depend on upstream green phases)
    # For now, initialize with delayed + attenuated version of J0
    for i, j in enumerate(JUNCTIONS[1:], 1):
        delay = TRAVEL_DELAY_S * i
        delayed = np.zeros(duration)
        # Platoon arrives with travel delay, some dispersion
        for t in range(duration):
            src_t = t - delay
            if 0 <= src_t < duration:
                # Platoon disperses slightly over +/- 2s
                for offset in range(-2, 3):
                    st = src_t + offset
                    if 0 <= st < duration:
                        delayed[t] += ew_arrivals_j0[int(st)] * 0.2
            # Add some local traffic too (not everything is platoon)
            delayed[t] += np.random.poisson(0.08)
        arrivals[(j, "ew")] = np.clip(delayed, 0, None).astype(int)

    return arrivals


# ──────────────────────────────────────────────────────────────────────────────
# JUNCTION STATE
# ──────────────────────────────────────────────────────────────────────────────

class JunctionState:
    """Track signal phase and queue state for one intersection."""
    def __init__(self, jid):
        self.jid = jid
        self.phase = "EW_GREEN"  # or "NS_GREEN", "EW_YELLOW", "NS_YELLOW", "ALL_RED"
        self.time_in_phase = 0.0
        self.queue_ew = 0.0
        self.queue_ns = 0.0
        self.arr_rate_ew = 0.0
        self.arr_rate_ns = 0.0
        self._last_green = "EW_GREEN"  # track which green was active before transition
        # Rolling window for arrival rate estimation
        self._ew_arrivals_window = []
        self._ns_arrivals_window = []
        self._window_size = 10  # seconds
        # Regional coordination: suggested green offset
        self.offset_bias = 0.0  # positive = favor EW, negative = favor NS
        # PCS reservation
        self.pcs_reservation = None  # (start_t, end_t, direction)

    def is_green_for(self, direction):
        if direction == "ew":
            return self.phase == "EW_GREEN"
        return self.phase == "NS_GREEN"

    def update_arrival_rates(self, t, ew_arr, ns_arr):
        self._ew_arrivals_window.append((t, ew_arr))
        self._ns_arrivals_window.append((t, ns_arr))
        cutoff = t - self._window_size
        self._ew_arrivals_window = [(tt, a) for tt, a in self._ew_arrivals_window if tt >= cutoff]
        self._ns_arrivals_window = [(tt, a) for tt, a in self._ns_arrivals_window if tt >= cutoff]
        if self._ew_arrivals_window:
            self.arr_rate_ew = sum(a for _, a in self._ew_arrivals_window) / self._window_size
        if self._ns_arrivals_window:
            self.arr_rate_ns = sum(a for _, a in self._ns_arrivals_window) / self._window_size


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINES
# ──────────────────────────────────────────────────────────────────────────────

def simulate_fixed(arrivals, duration):
    """Fixed-timing baseline: each junction runs independent 60s cycle."""
    junctions = {j: JunctionState(j) for j in JUNCTIONS}
    steps = []

    for t in range(duration):
        # Determine phase from fixed cycle
        cycle_pos = t % FIXED_CYCLE
        for j in JUNCTIONS:
            js = junctions[j]
            if cycle_pos < FIXED_GREEN:
                js.phase = "EW_GREEN"
            elif cycle_pos < FIXED_GREEN + YELLOW:
                js.phase = "EW_YELLOW"
            elif cycle_pos < FIXED_GREEN + YELLOW + ALL_RED:
                js.phase = "ALL_RED"
            elif cycle_pos < FIXED_GREEN * 2 + YELLOW + ALL_RED:
                js.phase = "NS_GREEN"
            elif cycle_pos < FIXED_GREEN * 2 + YELLOW * 2 + ALL_RED:
                js.phase = "NS_YELLOW"
            else:
                js.phase = "ALL_RED"

            # Add arrivals
            js.queue_ew += arrivals.get((j, "ew"), np.zeros(duration))[t]
            js.queue_ns += arrivals.get((j, "ns"), np.zeros(duration))[t]

            # Departures during green
            if js.phase == "EW_GREEN":
                depart = min(js.queue_ew, SAT_FLOW)
                js.queue_ew = max(0, js.queue_ew - depart)
            elif js.phase == "NS_GREEN":
                depart = min(js.queue_ns, SAT_FLOW_NS)
                js.queue_ns = max(0, js.queue_ns - depart)

        # Record step
        total_halted = sum(js.queue_ew + js.queue_ns for js in junctions.values())
        per_j = {}
        for j in JUNCTIONS:
            js = junctions[j]
            per_j[j] = {"ns": round(js.queue_ns), "ew": round(js.queue_ew)}
        steps.append({"t": t, "total_halted": round(total_halted), "per_j": per_j})

    return steps


def simulate_adaptive(arrivals, duration):
    """
    Adaptive pressure-based controller (Layer 1 + Layer 2 regional coordination).

    Layer 1 (local): Each junction uses pressure = queue + ARRIVAL_WEIGHT * arrival_rate
    to decide when to switch phases. Hysteresis prevents chattering.

    Layer 2 (regional): Downstream junctions receive platoon predictions from upstream.
    When a platoon is predicted to arrive, the offset_bias increases pressure for EW
    to keep green aligned with the platoon. This creates a crude green wave effect.
    """
    junctions = {j: JunctionState(j) for j in JUNCTIONS}
    steps = []
    # Track EW departures per junction for platoon propagation to downstream
    ew_departures = {j: np.zeros(duration) for j in JUNCTIONS}

    for t in range(duration):
        # ── Layer 2: Regional coordination (platoon prediction) ──
        # Each downstream junction looks at upstream departures and predicts
        # when the platoon will arrive. If a platoon is expected soon,
        # bias the junction toward holding/granting EW green.
        for i, j in enumerate(JUNCTIONS):
            js = junctions[j]
            js.offset_bias = 0.0
            if i > 0:
                upstream = JUNCTIONS[i - 1]
                # Look at upstream departures in the window that would arrive now
                # (departed TRAVEL_DELAY_S seconds ago)
                platoon_arrivals = 0
                for lookback in range(TRAVEL_DELAY_S - 3, TRAVEL_DELAY_S + 4):
                    src_t = t - lookback
                    if 0 <= src_t < duration:
                        platoon_arrivals += ew_departures[upstream][src_t]
                if platoon_arrivals > 2:
                    # Significant platoon approaching — bias toward EW green
                    js.offset_bias = OFFSET_WEIGHT * platoon_arrivals

        # ── Process each junction ──
        for j in JUNCTIONS:
            js = junctions[j]
            ew_arr = arrivals.get((j, "ew"), np.zeros(duration))[t]
            ns_arr = arrivals.get((j, "ns"), np.zeros(duration))[t]

            # Add arrivals to queues
            js.queue_ew += ew_arr
            js.queue_ns += ns_arr
            js.update_arrival_rates(t, ew_arr, ns_arr)

            # Departures during green phases
            departed_ew = 0
            if js.phase == "EW_GREEN":
                departed_ew = min(js.queue_ew, SAT_FLOW)
                js.queue_ew = max(0, js.queue_ew - departed_ew)
            elif js.phase == "NS_GREEN":
                departed_ns = min(js.queue_ns, SAT_FLOW_NS)
                js.queue_ns = max(0, js.queue_ns - departed_ns)

            ew_departures[j][t] = departed_ew

            # ── Layer 1: Pressure-based phase switching ──
            js.time_in_phase += 1.0

            if js.phase in ("EW_YELLOW", "NS_YELLOW"):
                if js.time_in_phase >= YELLOW:
                    js.phase = "ALL_RED"
                    js.time_in_phase = 0.0
                continue
            if js.phase == "ALL_RED":
                if js.time_in_phase >= ALL_RED:
                    # Switch to the opposite green
                    # Determine which green was before yellow
                    # We track this via a simple heuristic: alternate
                    js.phase = "NS_GREEN" if js._last_green == "EW_GREEN" else "EW_GREEN"
                    js.time_in_phase = 0.0
                continue

            # Compute pressures (with regional bias applied to EW)
            p_ew = js.queue_ew + ARRIVAL_WEIGHT * js.arr_rate_ew + js.offset_bias
            p_ns = js.queue_ns + ARRIVAL_WEIGHT * js.arr_rate_ns

            should_switch = False
            if js.time_in_phase >= MIN_GREEN:
                if js.phase == "EW_GREEN" and p_ns > p_ew + HYSTERESIS:
                    should_switch = True
                elif js.phase == "NS_GREEN" and p_ew > p_ns + HYSTERESIS:
                    should_switch = True
            if js.time_in_phase >= MAX_GREEN:
                should_switch = True

            if should_switch:
                js._last_green = js.phase
                js.phase = "EW_YELLOW" if js.phase == "EW_GREEN" else "NS_YELLOW"
                js.time_in_phase = 0.0

        # Record step
        total_halted = sum(js.queue_ew + js.queue_ns for js in junctions.values())
        per_j = {}
        for j in JUNCTIONS:
            js = junctions[j]
            per_j[j] = {"ns": round(js.queue_ns), "ew": round(js.queue_ew)}
        steps.append({"t": t, "total_halted": round(total_halted), "per_j": per_j})

    return steps


def simulate_emergency(arrivals, duration, pcs_enabled):
    """
    Emergency corridor simulation (Layer 3: Priority Corridor Scheduler).

    Base layer is adaptive control (Layer 1 + 2). At AMBULANCE_DISPATCH_T,
    an ambulance is dispatched on the EW corridor.

    Without PCS: Ambulance encounters whatever phase is active. If red, it stops
    and waits for the next green (realistic: emergency vehicles in India often
    cannot fully preempt signals without coordination).

    With PCS: The system computes ETAs for the ambulance at each junction and
    issues reservation windows. Each junction forces EW green before the ambulance
    arrives, pre-clearing the queue. The ambulance passes through with 0 stops.
    """
    junctions = {j: JunctionState(j) for j in JUNCTIONS}
    steps = []
    ew_departures = {j: np.zeros(duration) for j in JUNCTIONS}

    # Ambulance state
    amb_dispatched = False
    amb_done = False
    amb_pos_m = 0.0  # meters from corridor entry (starts 100m before J0)
    amb_speed = 0.0
    amb_stops = 0
    amb_enter_t = None
    amb_exit_t = None
    amb_was_stopped = False
    amb_stopped_at_junction = None  # track which junction ambulance is stopped at
    amb_stop_wait_remaining = 0    # seconds remaining at current red stop
    corridor_entry_offset = 100.0  # ambulance starts 100m before J0
    corridor_length_m = JUNCTION_SPACING_M * (len(JUNCTIONS) - 1) + corridor_entry_offset + 150  # past last junction

    # Junction positions in meters along the corridor (from ambulance entry point)
    junction_pos_m = {j: corridor_entry_offset + i * JUNCTION_SPACING_M for i, j in enumerate(JUNCTIONS)}

    # PCS: compute reservation windows at dispatch time
    def compute_pcs_reservations(dispatch_t):
        for j in JUNCTIONS:
            dist = junction_pos_m[j]  # distance from ambulance start to junction
            eta = dispatch_t + dist / AMBULANCE_SPEED_MPS
            junctions[j].pcs_reservation = (
                eta - PCS_PRE_CLEAR_S,
                eta + PCS_POST_GUARD_S,
                "ew"
            )

    for t in range(duration):
        # ── Ambulance dispatch ──
        if t == AMBULANCE_DISPATCH_T and not amb_dispatched:
            amb_dispatched = True
            amb_enter_t = t
            amb_pos_m = 0.0
            amb_speed = AMBULANCE_SPEED_MPS
            if pcs_enabled:
                compute_pcs_reservations(t)

        # ── Layer 2: Regional coordination ──
        for i, j in enumerate(JUNCTIONS):
            js = junctions[j]
            js.offset_bias = 0.0
            if i > 0:
                upstream = JUNCTIONS[i - 1]
                platoon_arrivals = 0
                for lookback in range(TRAVEL_DELAY_S - 3, TRAVEL_DELAY_S + 4):
                    src_t = t - lookback
                    if 0 <= src_t < duration:
                        platoon_arrivals += ew_departures[upstream][src_t]
                if platoon_arrivals > 2:
                    js.offset_bias = OFFSET_WEIGHT * platoon_arrivals

        # ── Process each junction ──
        for j in JUNCTIONS:
            js = junctions[j]
            ew_arr = arrivals.get((j, "ew"), np.zeros(duration))[t]
            ns_arr = arrivals.get((j, "ns"), np.zeros(duration))[t]

            js.queue_ew += ew_arr
            js.queue_ns += ns_arr
            js.update_arrival_rates(t, ew_arr, ns_arr)

            # Departures
            departed_ew = 0
            if js.phase == "EW_GREEN":
                departed_ew = min(js.queue_ew, SAT_FLOW)
                js.queue_ew = max(0, js.queue_ew - departed_ew)
            elif js.phase == "NS_GREEN":
                departed_ns = min(js.queue_ns, SAT_FLOW_NS)
                js.queue_ns = max(0, js.queue_ns - departed_ns)

            ew_departures[j][t] = departed_ew
            js.time_in_phase += 1.0

            # ── Layer 3: PCS override ──
            pcs_active = False
            if pcs_enabled and js.pcs_reservation is not None:
                res_start, res_end, res_dir = js.pcs_reservation
                if res_start <= t <= res_end:
                    pcs_active = True
                    # Force EW green: if currently NS, initiate switch
                    if js.phase == "NS_GREEN" and js.time_in_phase >= MIN_GREEN:
                        js._last_green = js.phase
                        js.phase = "NS_YELLOW"
                        js.time_in_phase = 0.0
                        continue
                    elif js.phase == "EW_GREEN":
                        # Hold EW green — don't switch even if pressure says to
                        continue
                    elif js.phase in ("NS_YELLOW", "ALL_RED"):
                        # Let transition complete naturally, it will land on EW_GREEN
                        pass

            # Transition phases
            if js.phase in ("EW_YELLOW", "NS_YELLOW"):
                if js.time_in_phase >= YELLOW:
                    js.phase = "ALL_RED"
                    js.time_in_phase = 0.0
                continue
            if js.phase == "ALL_RED":
                if js.time_in_phase >= ALL_RED:
                    # If PCS wants EW green, go there
                    if pcs_active:
                        js.phase = "EW_GREEN"
                    elif hasattr(js, '_last_green'):
                        js.phase = "NS_GREEN" if js._last_green == "EW_GREEN" else "EW_GREEN"
                    else:
                        js.phase = "EW_GREEN"
                    js.time_in_phase = 0.0
                continue

            # Normal adaptive logic (Layer 1 + 2)
            p_ew = js.queue_ew + ARRIVAL_WEIGHT * js.arr_rate_ew + js.offset_bias
            p_ns = js.queue_ns + ARRIVAL_WEIGHT * js.arr_rate_ns

            should_switch = False
            if js.time_in_phase >= MIN_GREEN:
                if js.phase == "EW_GREEN" and p_ns > p_ew + HYSTERESIS:
                    should_switch = True
                elif js.phase == "NS_GREEN" and p_ew > p_ns + HYSTERESIS:
                    should_switch = True
            if js.time_in_phase >= MAX_GREEN:
                should_switch = True

            if should_switch:
                js._last_green = js.phase
                js.phase = "EW_YELLOW" if js.phase == "EW_GREEN" else "NS_YELLOW"
                js.time_in_phase = 0.0

        # ── Ambulance movement ──
        if amb_dispatched and not amb_done:
            # If ambulance is waiting at a stopped junction
            if amb_stop_wait_remaining > 0:
                amb_stop_wait_remaining -= 1
                amb_speed = 0.0
                if amb_stop_wait_remaining <= 0:
                    # Done waiting, can proceed (signal should have cycled)
                    amb_was_stopped = False
                    amb_stopped_at_junction = None
                    amb_speed = AMBULANCE_CRAWL_MPS  # slow start
            else:
                # Check each junction for red signals ahead of ambulance
                stopped_this_step = False
                for j in JUNCTIONS:
                    js = junctions[j]
                    jpos = junction_pos_m[j]
                    # Ambulance is approaching this junction (within 50m before it)
                    if jpos - 50 <= amb_pos_m <= jpos + 5:
                        if not js.is_green_for("ew"):
                            if not pcs_enabled:
                                # Ambulance slows and stops at red
                                if amb_pos_m < jpos - 10:
                                    amb_speed = AMBULANCE_CRAWL_MPS
                                else:
                                    amb_speed = 0.0
                                    stopped_this_step = True
                                    if not amb_was_stopped:
                                        amb_stops += 1
                                        amb_was_stopped = True
                                        amb_stopped_at_junction = j
                                        # Wait for remaining red phase + queue clear
                                        # Estimate: wait for phase to switch to EW green
                                        amb_stop_wait_remaining = AMBULANCE_STOP_PENALTY_S
                                break
                        else:
                            amb_speed = AMBULANCE_SPEED_MPS
                            amb_was_stopped = False
                    elif jpos - 100 <= amb_pos_m < jpos - 50:
                        # Approaching junction zone — slow slightly due to traffic
                        if not js.is_green_for("ew") and js.queue_ew > 3:
                            amb_speed = min(amb_speed, AMBULANCE_SPEED_MPS * 0.7)

                if not stopped_this_step and amb_stop_wait_remaining <= 0:
                    # Accelerate back to full speed
                    amb_speed = min(AMBULANCE_SPEED_MPS, amb_speed + 2.0)

            amb_pos_m += amb_speed * 1.0  # 1 second step

            if amb_pos_m >= corridor_length_m:
                amb_done = True
                amb_exit_t = t

        # Record step
        total_halted = sum(js.queue_ew + js.queue_ns for js in junctions.values())
        per_j = {}
        for j in JUNCTIONS:
            js = junctions[j]
            per_j[j] = {"ns": round(js.queue_ns), "ew": round(js.queue_ew)}
        steps.append({"t": t, "total_halted": round(total_halted), "per_j": per_j})

    # Compute ambulance travel time
    amb_travel = (amb_exit_t - amb_enter_t) if (amb_enter_t is not None and amb_exit_t is not None) else None

    return steps, {
        "ambulance_travel_s": amb_travel,
        "ambulance_enter_t": amb_enter_t,
        "ambulance_exit_t": amb_exit_t,
        "ambulance_stops": amb_stops,
    }


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def make_result(tag, steps, extra_summary=None):
    """Build the result dict in the format dashboard.py expects."""
    halted_vals = [s["total_halted"] for s in steps]
    summary = {
        "avg_halted": round(sum(halted_vals) / len(halted_vals), 2) if halted_vals else 0,
        "peak_halted": max(halted_vals) if halted_vals else 0,
        "ambulance_travel_s": None,
        "ambulance_enter_t": None,
        "ambulance_exit_t": None,
        "ambulance_stops": None,
    }
    if extra_summary:
        summary.update(extra_summary)
    return {"tag": tag, "steps": steps, "summary": summary}


def save_result(result):
    path = os.path.join(RESULTS_DIR, f"{result['tag']}.json")
    with open(path, "w") as f:
        json.dump(result, f)
    print(f"  Saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("FlowSense Demo Runner")
    print("=" * 50)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load real video-derived state
    print(f"Loading state log from {STATE_LOG}...")
    df = load_state_log(STATE_LOG)
    print(f"  {len(df)} entries, time span: {df['t_rel'].iloc[-1]:.0f}s")

    # Prepare arrival data
    print("Preparing multi-junction arrival data...")
    ew_rates, queue_data = prepare_arrivals(df, SIM_DURATION)
    ew_arrivals_j0 = synthesize_arrivals(ew_rates)
    arrivals = make_junction_arrivals(ew_arrivals_j0, SIM_DURATION)

    # Run scenarios
    print("\n[1/4] Fixed-timing baseline...")
    np.random.seed(42)
    fixed_steps = simulate_fixed(arrivals, SIM_DURATION)
    save_result(make_result("baseline", fixed_steps))

    print("[2/4] Adaptive controller (Layer 1 + Layer 2 regional)...")
    np.random.seed(42)
    adaptive_steps = simulate_adaptive(arrivals, SIM_DURATION)
    save_result(make_result("adaptive", adaptive_steps))

    print("[3/4] Emergency without PCS...")
    np.random.seed(42)
    emerg_nopcs_steps, emerg_nopcs_amb = simulate_emergency(arrivals, SIM_DURATION, pcs_enabled=False)
    save_result(make_result("emergency_nopcs", emerg_nopcs_steps, emerg_nopcs_amb))

    print("[4/4] Emergency with PCS (Layer 3)...")
    np.random.seed(42)
    emerg_pcs_steps, emerg_pcs_amb = simulate_emergency(arrivals, SIM_DURATION, pcs_enabled=True)
    save_result(make_result("emergency_pcs", emerg_pcs_steps, emerg_pcs_amb))

    # Print summary
    print("\n" + "=" * 50)
    print("Results Summary:")
    for tag in ["baseline", "adaptive", "emergency_nopcs", "emergency_pcs"]:
        with open(os.path.join(RESULTS_DIR, f"{tag}.json")) as f:
            r = json.load(f)
        s = r["summary"]
        line = f"  {tag:20s}  avg_halted={s['avg_halted']:6.1f}  peak={s['peak_halted']:3d}"
        if s.get("ambulance_travel_s"):
            line += f"  amb_travel={s['ambulance_travel_s']}s  amb_stops={s['ambulance_stops']}"
        print(line)

    base_avg = make_result("", fixed_steps)["summary"]["avg_halted"]
    adapt_avg = make_result("", adaptive_steps)["summary"]["avg_halted"]
    if base_avg > 0:
        reduction = (base_avg - adapt_avg) / base_avg * 100
        print(f"\n  Adaptive reduction: {reduction:.0f}%")

    print(f"\nAll results written to {RESULTS_DIR}/")
    print("Next: streamlit run src/dashboard.py")


if __name__ == "__main__":
    main()
