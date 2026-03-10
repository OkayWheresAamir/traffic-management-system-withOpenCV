# src/controller.py
# Improved visual layout + debug for 2-approach adaptive controller (Road A = estimator, Road B = simulated)
import time, json, os, random
import numpy as np
import cv2

# ---------- CONFIG ----------
STATE_LOG = "output/state_log.jsonl"   # written by state_estimator.py
SIGNAL_LOG = "output/signal_log.jsonl" # controller decisions + timestamps

MIN_GREEN = 10.0
MAX_GREEN = 45.0
YELLOW = 3.0
ALL_RED = 1.0
EVAL_INTERVAL = 0.5

ARRIVAL_WEIGHT = 5.0
HYSTERESIS = 1.0

# Road B (simulated)
LAMBDA_B_PER_MIN = 30.0    # expected vehicles/minute on Road B
SATURATION_FLOW_B = 1.0    # veh/s while green (approx)

# Visualization
WIN_W, WIN_H = 900, 360
BG = (30, 30, 30)
TEXT = (230, 230, 230)

# ---------- Helpers ----------
def tail_latest(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            lines = f.read().strip().splitlines()
            if not lines:
                return None
            return json.loads(lines[-1])
    except Exception:
        return None

def write_signal_log(entry):
    with open(SIGNAL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ---------- State ----------
phase = "A_GREEN"   # "A_GREEN" or "B_GREEN"
phase_start = time.time()

queue_A = 0
arrival_rate_A = 0.0
queue_B = 0
arrival_times_B = []
WINDOW = 10.0
lambda_B_per_sec = LAMBDA_B_PER_MIN / 60.0

last_eval = time.time()

print("Controller started. Reading", STATE_LOG)

# ---------- Main loop ----------
while True:
    loop_start = time.time()

    # read latest state
    state = tail_latest(STATE_LOG)
    if state is not None:
        queue_A = float(state.get("queue_length", queue_A))
        arrival_rate_A = float(state.get("arrival_rate", arrival_rate_A))
        # departure_rate_A not used for simulation of A (we only observe it)
    else:
        # if no state yet, keep previous values (controller waits)
        pass

    # DEBUG: show what we read
    print(f"[DBG] Read state: queue_A={queue_A} arrA={arrival_rate_A:.2f}")

    # Simulate Road B arrivals (Poisson for this interval)
    now = time.time()
    mean = lambda_B_per_sec * EVAL_INTERVAL
    arrivals_B = np.random.poisson(mean)
    if arrivals_B > 0:
        queue_B += int(arrivals_B)
        for _ in range(arrivals_B):
            arrival_times_B.append(now)

    # prune arrival_times_B for WINDOW
    cutoff = now - WINDOW
    arrival_times_B = [t for t in arrival_times_B if t >= cutoff]
    arrival_rate_B = len(arrival_times_B) / WINDOW

    # discharge on B if B has green
    if phase == "B_GREEN":
        # how many vehicles can depart this interval
        can_depart = SATURATION_FLOW_B * EVAL_INTERVAL
        departed = min(queue_B, int(round(can_depart)))
        queue_B -= departed
        # note: we don't need to use departure rate for control right now

    # compute pressures
    pressure_A = queue_A + arrival_rate_A * ARRIVAL_WEIGHT
    pressure_B = queue_B + arrival_rate_B * ARRIVAL_WEIGHT

    # decision logic
    time_in_phase = time.time() - phase_start
    decision = "HOLD"
    if time_in_phase >= MIN_GREEN:
        if phase == "A_GREEN" and pressure_B > pressure_A + HYSTERESIS:
            decision = "SWITCH_TO_B"
        elif phase == "B_GREEN" and pressure_A > pressure_B + HYSTERESIS:
            decision = "SWITCH_TO_A"
    if time_in_phase >= MAX_GREEN:
        decision = "SWITCH_TO_B" if phase == "A_GREEN" else "SWITCH_TO_A"

    # execute
    if decision.startswith("SWITCH"):
        print(f"[CTRL] {phase} -> switching ({decision})")
        write_signal_log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time())),
                          "event": "switch_start", "from": phase, "to": ("B_GREEN" if phase=="A_GREEN" else "A_GREEN")})
        time.sleep(YELLOW)
        time.sleep(ALL_RED)
        phase = "B_GREEN" if phase == "A_GREEN" else "A_GREEN"
        phase_start = time.time()
        write_signal_log({"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time())),
                          "event": "switch_complete", "phase": phase})

    # periodic log of status
    status = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time())),
        "phase": phase,
        "time_in_phase": round(time.time() - phase_start, 1),
        "queue_A": int(queue_A),
        "arrA": round(arrival_rate_A, 3),
        "queue_B": int(queue_B),
        "arrB": round(arrival_rate_B, 3),
        "P_A": round(pressure_A,2),
        "P_B": round(pressure_B,2)
    }
    write_signal_log(status)

    # ---------- Visualization (clean layout) ----------
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = BG

    # Phase big circle
    if phase == "A_GREEN":
        phase_col = (50,200,50)
    else:
        phase_col = (50,50,200)
    cv2.circle(canvas, (80, 80), 40, phase_col, -1)
    cv2.putText(canvas, f"{phase}", (140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2)

    # Draw Road A block (left)
    # Bar area
    base_xA = 40; base_yA = 180; bar_w = 100; bar_h = 140
    cv2.rectangle(canvas, (base_xA-5, base_yA-5), (base_xA+bar_w+5, base_yA+bar_h+5), (50,50,50), -1)
    # Fill A bar
    max_q = 50.0
    hA = int(min(bar_h, (queue_A/max_q) * bar_h))
    cv2.rectangle(canvas, (base_xA, base_yA+bar_h-hA), (base_xA+bar_w, base_yA+bar_h), (50,180,50) if phase=="A_GREEN" else (80,80,80), -1)
    cv2.putText(canvas, f"Road A", (base_xA+bar_w+20, base_yA+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)
    cv2.putText(canvas, f"q={int(queue_A)}", (base_xA+bar_w+20, base_yA+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)
    cv2.putText(canvas, f"arr/s={arrival_rate_A:.2f}", (base_xA+bar_w+20, base_yA+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)
    cv2.putText(canvas, f"P={pressure_A:.1f}", (base_xA+bar_w+20, base_yA+110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)

    # Draw Road B block (right)
    base_xB = 320; base_yB = 180
    cv2.rectangle(canvas, (base_xB-5, base_yB-5), (base_xB+bar_w+5, base_yB+bar_h+5), (50,50,50), -1)
    hB = int(min(bar_h, (queue_B/max_q) * bar_h))
    cv2.rectangle(canvas, (base_xB, base_yB+bar_h-hB), (base_xB+bar_w, base_yB+bar_h), (50,180,50) if phase=="B_GREEN" else (80,80,80), -1)
    cv2.putText(canvas, f"Road B (sim)", (base_xB+bar_w+20, base_yB+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)
    cv2.putText(canvas, f"q={int(queue_B)}", (base_xB+bar_w+20, base_yB+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)
    cv2.putText(canvas, f"arr/s={arrival_rate_B:.2f}", (base_xB+bar_w+20, base_yB+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)
    cv2.putText(canvas, f"P={pressure_B:.1f}", (base_xB+bar_w+20, base_yB+110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)

    # bottom row: decision and timers
    cv2.putText(canvas, f"Decision: {decision}", (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 2)
    cv2.putText(canvas, f"TimeInPhase: {int(time.time()-phase_start)}s", (base_xB+bar_w+20, base_yB+140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1)

    cv2.imshow("Adaptive Controller Sim (clean)", canvas)

    # wait; keep UI interactive
    if cv2.waitKey(int(EVAL_INTERVAL*1000)) & 0xFF == 27:
        break

# cleanup
cv2.destroyAllWindows()
print("Controller stopped. Signal log:", SIGNAL_LOG)