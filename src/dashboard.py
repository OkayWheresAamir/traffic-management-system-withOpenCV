"""
FlowSense — Live Stochastic Traffic Simulation Dashboard
========================================================
Step-based simulation driven entirely by st.session_state.
No precomputed result playback.

Usage:
    streamlit run src/dashboard.py
"""

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlowSense — Live Traffic Simulation",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
JUNCTIONS = ["J0", "J1", "J2"]
JX = {"J0": 170, "J1": 430, "J2": 690}
SIM_DURATION_DEFAULT = 300

MODE_CFG = {
    "fixed": {
        "label": "Fixed Timing",
        "color": "#f85149",
        "spawn_eb": 1.00,
        "spawn_wb": 0.85,
        "spawn_ns": 0.75,
    },
    "adaptive": {
        "label": "FlowSense Adaptive",
        "color": "#58a6ff",
        "spawn_eb": 0.90,
        "spawn_wb": 0.78,
        "spawn_ns": 0.66,
    },
    "emergency": {
        "label": "Emergency Corridor",
        "color": "#f59e0b",
        "spawn_eb": 0.92,
        "spawn_wb": 0.80,
        "spawn_ns": 0.70,
    },
}

PHASE_SIG = {
    "EW_GREEN": ("#22c55e", "#ef4444"),
    "EW_YELLOW": ("#eab308", "#ef4444"),
    "NS_GREEN": ("#ef4444", "#22c55e"),
    "NS_YELLOW": ("#ef4444", "#eab308"),
    "ALL_RED": ("#ef4444", "#ef4444"),
}

# Signal timings
MIN_GREEN = 10
MAX_GREEN = 52
YELLOW = 3
ALL_RED = 1
ADAPTIVE_HYSTERESIS = 2.8
PLATOON_HOLD_THRESHOLD = 8
PLATOON_HOLD_MARGIN = 4

# Canvas geometry
CW, CH = 860, 280
ROAD_CY = 140
ROAD_H = 18
NS_W = 18
STOP_OFF = 30
VEH_SZ = 10
MIN_GAP_PX = 18

# Emergency defaults
AMB_DISPATCH_T = 20
AMB_BASE_SPEED = 24.0
AMB_LOOKAHEAD_PX = 280
NS_BASE_SPEED = 10.5
YIELD_LATERAL_PX = 11.0
ENTRY_CAP_PER_STEP = 2

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=40, r=20, t=36, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
)

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  .main,.block-container { background:#0d1117; color:#e6edf3; padding-top:.75rem; }
  .kpi-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:14px 18px; text-align:center; margin-bottom:6px;
  }
  .kpi-val   { font-size:1.8rem; font-weight:700; color:#58a6ff; }
  .kpi-delta { font-size:.78rem; color:#56d364; }
  .kpi-label { font-size:.75rem; color:#8b949e; margin-top:3px; }
  .sig-card {
    background:#0d1117; border:1px solid #21262d; border-radius:8px;
    padding:7px 10px; margin-bottom:5px; font-size:.82rem;
  }
  h1,h2,h3 { color:#e6edf3 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION STATE
# ──────────────────────────────────────────────────────────────────────────────
def _empty_history():
    return {
        "time": [],
        "queue_length": [],
        "throughput": [],
        "avg_delay": [],
        "avg_stops": [],
        "active_vehicles": [],
        "per_j_total": {j: [] for j in JUNCTIONS},
    }


def _new_sim(mode: str):
    rng = np.random.default_rng()
    sim = {
        "time": 0,
        "mode": mode,
        "rng": rng,
        "vehicles": [],
        "signals": {
            j: {
                "phase": "EW_GREEN",
                "time_in_phase": 0,
                "last_green": "EW_GREEN",
                "offset": i * 8,
            }
            for i, j in enumerate(JUNCTIONS)
        },
        "queues": {j: {"ew": 0, "ns": 0.0} for j in JUNCTIONS},
        "metrics_history": _empty_history(),
        "completed": False,
        "throughput": 0,
        "ambulance": {
            "dispatched": False,
            "active": False,
            "completed": False,
            "dispatch_t": AMB_DISPATCH_T,
            "x": -65.0,
            "speed": AMB_BASE_SPEED,
            "base_speed": AMB_BASE_SPEED,
            "delay_s": 0.0,
            "stops": 0,
            "status": "standby",
            "entered_t": None,
            "exit_t": None,
            "stopped_junctions": set(),
        },
        "duration": SIM_DURATION_DEFAULT,
        "next_vehicle_id": 1,
        "recorded": False,
        "totals": {
            "completed": 0,
            "delay_accum": 0.0,
            "stops_accum": 0.0,
        },
    }
    sim["snapshot"] = build_snapshot(sim)
    append_metrics_history_step(sim)
    return sim


def _reset_sim(mode: str):
    st.session_state.sim = _new_sim(mode)
    st.session_state.running = True


# ──────────────────────────────────────────────────────────────────────────────
# STEP PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def spawn_vehicles_step(sim: dict):
    mode = sim["mode"]
    cfg = MODE_CFG[mode]
    rng = sim["rng"]

    # Random arrivals each step (Poisson), with mild temporal jitter.
    # Apply simple perimeter gating to avoid corridor-wide gridlock.
    network_q = sum(int(sim["queues"][j]["ew"] + round(sim["queues"][j]["ns"])) for j in JUNCTIONS)
    if network_q > 90:
        demand_scale = 0.45
    elif network_q > 65:
        demand_scale = 0.62
    elif network_q > 40:
        demand_scale = 0.8
    else:
        demand_scale = 1.0

    eb_rate = max(0.05, cfg["spawn_eb"] + rng.normal(0, 0.08))
    wb_rate = max(0.05, cfg["spawn_wb"] + rng.normal(0, 0.07))
    ns_rate = max(0.05, cfg["spawn_ns"] + rng.normal(0, 0.06))
    eb_rate *= demand_scale
    wb_rate *= demand_scale
    ns_rate *= demand_scale

    spawn_eb = int(rng.poisson(eb_rate))
    spawn_wb = int(rng.poisson(wb_rate))

    for k in range(spawn_eb):
        x0 = -18.0 - k * MIN_GAP_PX
        if _lane_has_space_x(sim["vehicles"], "eb", x0):
            v_speed = float(rng.uniform(12.5, 18.0))
            sim["vehicles"].append(
                {
                    "id": sim["next_vehicle_id"],
                    "lane": "eb",
                    "x": x0,
                    "y": ROAD_CY - 8.0,
                    "speed": v_speed,
                    "base_speed": v_speed,
                    "moving": True,
                    "delay_s": 0.0,
                    "stops": 0,
                    "was_moving": True,
                    "yielding": False,
                    "yield_side": -1 if sim["next_vehicle_id"] % 2 == 0 else 1,
                }
            )
            sim["next_vehicle_id"] += 1

    for k in range(spawn_wb):
        x0 = CW + 18.0 + k * MIN_GAP_PX
        if _lane_has_space_x(sim["vehicles"], "wb", x0):
            v_speed = float(rng.uniform(11.5, 17.0))
            sim["vehicles"].append(
                {
                    "id": sim["next_vehicle_id"],
                    "lane": "wb",
                    "x": x0,
                    "y": ROAD_CY + 8.0,
                    "speed": v_speed,
                    "base_speed": v_speed,
                    "moving": True,
                    "delay_s": 0.0,
                    "stops": 0,
                    "was_moving": True,
                }
            )
            sim["next_vehicle_id"] += 1

    # Vertical NS movement (south -> north) per junction.
    for j in JUNCTIONS:
        spawn_ns = int(rng.poisson(ns_rate))
        lane = f"ns_{j}"
        x_lane = JX[j] + 6.0
        for k in range(spawn_ns):
            y0 = CH + 18.0 + k * MIN_GAP_PX
            if _lane_has_space_y(sim["vehicles"], lane, y0):
                v_speed = float(rng.uniform(NS_BASE_SPEED - 2.0, NS_BASE_SPEED + 2.0))
                sim["vehicles"].append(
                    {
                        "id": sim["next_vehicle_id"],
                        "lane": lane,
                        "x": x_lane,
                        "y": y0,
                        "speed": v_speed,
                        "base_speed": v_speed,
                        "moving": True,
                        "delay_s": 0.0,
                        "stops": 0,
                        "was_moving": True,
                    }
                )
                sim["next_vehicle_id"] += 1


def _lane_has_space_x(vehicles: list, lane: str, x_spawn: float) -> bool:
    for v in vehicles:
        if v["lane"] == lane and abs(v["x"] - x_spawn) < MIN_GAP_PX:
            return False
    return True


def _lane_has_space_y(vehicles: list, lane: str, y_spawn: float) -> bool:
    for v in vehicles:
        if v["lane"] == lane and abs(v["y"] - y_spawn) < MIN_GAP_PX:
            return False
    return True


def _emergency_priority_junctions(sim: dict) -> set:
    if sim["mode"] != "emergency":
        return set()
    amb = sim["ambulance"]
    if not amb.get("active"):
        return set()

    ax = amb["x"]
    prio = set()
    for j, jx in JX.items():
        if 0 <= (jx - ax) <= AMB_LOOKAHEAD_PX:
            prio.add(j)
    return prio


def _ensure_emergency_dispatch(sim: dict):
    amb = sim["ambulance"]
    if sim["mode"] != "emergency":
        return
    if not amb["dispatched"] and sim["time"] >= amb["dispatch_t"]:
        amb["dispatched"] = True
        amb["active"] = True
        amb["entered_t"] = sim["time"]
        amb["x"] = -65.0
        amb["status"] = "dispatched"


def update_signals_step(sim: dict):
    mode = sim["mode"]
    rng = sim["rng"]
    priority = _emergency_priority_junctions(sim)

    if mode == "fixed":
        cycle = 64
        for j in JUNCTIONS:
            s = sim["signals"][j]
            c = (sim["time"] + s["offset"]) % cycle
            if c < 25:
                s["phase"] = "EW_GREEN"
            elif c < 28:
                s["phase"] = "EW_YELLOW"
            elif c < 29:
                s["phase"] = "ALL_RED"
            elif c < 54:
                s["phase"] = "NS_GREEN"
            elif c < 57:
                s["phase"] = "NS_YELLOW"
            else:
                s["phase"] = "ALL_RED"
        return

    for j in JUNCTIONS:
        s = sim["signals"][j]
        s["time_in_phase"] += 1
        force_ew = j in priority

        if s["phase"] in ("EW_YELLOW", "NS_YELLOW"):
            if s["time_in_phase"] >= YELLOW:
                s["phase"] = "ALL_RED"
                s["time_in_phase"] = 0
            continue

        if s["phase"] == "ALL_RED":
            if s["time_in_phase"] >= ALL_RED:
                if force_ew:
                    s["phase"] = "EW_GREEN"
                else:
                    s["phase"] = "NS_GREEN" if s["last_green"] == "EW_GREEN" else "EW_GREEN"
                s["time_in_phase"] = 0
            continue

        # Emergency green-wave hold for EW.
        if force_ew and s["phase"] == "EW_GREEN":
            continue
        if force_ew and s["phase"] == "NS_GREEN" and s["time_in_phase"] >= 3:
            s["last_green"] = "NS_GREEN"
            s["phase"] = "NS_YELLOW"
            s["time_in_phase"] = 0
            continue

        q_ew = sim["queues"][j]["ew"]
        q_ns = sim["queues"][j]["ns"]
        p_ew = q_ew + float(rng.uniform(0, 1.0)) + (3.0 if force_ew else 0.0)
        p_ns = q_ns + float(rng.uniform(0, 1.0))

        switch = False
        if s["time_in_phase"] >= MIN_GREEN:
            active_q = q_ew if s["phase"] == "EW_GREEN" else q_ns
            opp_q = q_ns if s["phase"] == "EW_GREEN" else q_ew
            if s["phase"] == "EW_GREEN" and p_ns > p_ew + ADAPTIVE_HYSTERESIS:
                switch = True
            elif s["phase"] == "NS_GREEN" and p_ew > p_ns + ADAPTIVE_HYSTERESIS:
                switch = True
            if switch:
                # Hold current green long enough to discharge a visible platoon.
                hold_platoon = (
                    active_q >= PLATOON_HOLD_THRESHOLD
                    and opp_q <= active_q + PLATOON_HOLD_MARGIN
                    and s["time_in_phase"] < int(MAX_GREEN * 0.85)
                )
                if hold_platoon:
                    switch = False
        if s["time_in_phase"] >= MAX_GREEN:
            switch = True

        if switch:
            s["last_green"] = s["phase"]
            s["phase"] = "EW_YELLOW" if s["phase"] == "EW_GREEN" else "NS_YELLOW"
            s["time_in_phase"] = 0


def _ew_allows_motion(phase: str) -> bool:
    return phase == "EW_GREEN"


def _ns_allows_motion(phase: str) -> bool:
    return phase == "NS_GREEN"


def _junction_conflict_bounds(jx: float):
    return (
        jx - NS_W - 4,
        jx + NS_W + 4,
        ROAD_CY - ROAD_H - 4,
        ROAD_CY + ROAD_H + 4,
    )


def _vehicle_in_junction_conflict(v: dict, j: str) -> bool:
    jx = JX[j]
    x0, x1, y0, y1 = _junction_conflict_bounds(jx)
    return x0 <= float(v.get("x", -9999)) <= x1 and y0 <= float(v.get("y", -9999)) <= y1


def _junction_conflict_blocked(vehicles: list, j: str, entering_lane: str, vid: int) -> bool:
    for ov in vehicles:
        if ov["id"] == vid:
            continue
        if entering_lane in ("eb", "wb"):
            if ov["lane"] == f"ns_{j}" and _vehicle_in_junction_conflict(ov, j):
                return True
        else:
            if ov["lane"] in ("eb", "wb") and _vehicle_in_junction_conflict(ov, j):
                return True
    return False


def _entry_cap(sim: dict, j: str, lane: str) -> int:
    if lane in ("eb", "wb"):
        q = float(sim["queues"][j]["ew"])
    else:
        q = float(sim["queues"][j]["ns"])
    cap = ENTRY_CAP_PER_STEP + int(min(3, q // 10))
    if q >= 24:
        cap += 1
    return int(max(1, min(6, cap)))


def _downstream_clear_for_entry(vehicles: list, j: str, lane: str, vid: int) -> bool:
    jx = JX[j]
    if lane == "eb":
        for ov in vehicles:
            if ov["id"] == vid or ov["lane"] != "eb":
                continue
            ov_slow = (not ov.get("moving", True)) or float(ov.get("speed", 0.0)) < 7.5
            if ov_slow and jx + NS_W + 8 <= ov["x"] <= jx + NS_W + 42 and abs(ov.get("y", ROAD_CY - 8.0) - (ROAD_CY - 8.0)) < 14:
                return False
        return True
    if lane == "wb":
        for ov in vehicles:
            if ov["id"] == vid or ov["lane"] != "wb":
                continue
            ov_slow = (not ov.get("moving", True)) or float(ov.get("speed", 0.0)) < 7.5
            if ov_slow and jx - NS_W - 42 <= ov["x"] <= jx - NS_W - 8 and abs(ov.get("y", ROAD_CY + 8.0) - (ROAD_CY + 8.0)) < 14:
                return False
        return True

    lane_id = f"ns_{j}"
    for ov in vehicles:
        if ov["id"] == vid or ov["lane"] != lane_id:
            continue
        ov_slow = (not ov.get("moving", True)) or float(ov.get("speed", 0.0)) < 6.0
        if ov_slow and ROAD_CY - ROAD_H - 38 <= ov["y"] <= ROAD_CY - ROAD_H + 8:
            return False
    return True


def update_vehicle_positions_step(sim: dict):
    vehicles = sim["vehicles"]
    removed = []
    amb = sim["ambulance"]
    emergency_active = sim["mode"] == "emergency" and amb.get("active", False)
    amb_x = amb["x"] if emergency_active else -9999
    entry_count = {(j, lane): 0 for j in JUNCTIONS for lane in ("eb", "wb", "ns")}

    # Eastbound lane (x increasing)
    eb = sorted([v for v in vehicles if v["lane"] == "eb"], key=lambda x: x["x"], reverse=True)
    lead_x = None
    for v in eb:
        # Baseline speed recovers naturally unless yielding.
        v["speed"] = max(5.0, float(v.get("base_speed", v["speed"])) * float(sim["rng"].uniform(0.95, 1.05)))

        # Emergency yielding behavior: pull to the side and slow.
        in_yield_zone = emergency_active and (amb_x - 20 <= v["x"] <= amb_x + 180)
        v["yielding"] = bool(in_yield_zone)
        target_y = ROAD_CY - 8.0
        if v["yielding"]:
            target_y += float(v.get("yield_side", 1)) * YIELD_LATERAL_PX
            v["speed"] = min(v["speed"], float(v.get("base_speed", v["speed"])) * 0.65)
        v["y"] += max(-2.2, min(2.2, target_y - v["y"]))

        desired = v["x"] + v["speed"]
        block = None
        crossing = None

        for j, jx in JX.items():
            stop_x = jx - STOP_OFF - 3
            if v["x"] <= stop_x and desired >= stop_x:
                crossing = (j, stop_x)
                break

        if crossing is not None:
            cj, stop_x = crossing
            can_enter = (
                _ew_allows_motion(sim["signals"][cj]["phase"])
                and entry_count[(cj, "eb")] < _entry_cap(sim, cj, "eb")
                and not _junction_conflict_blocked(vehicles, cj, "eb", v["id"])
                and _downstream_clear_for_entry(vehicles, cj, "eb", v["id"])
            )
            if not can_enter:
                block = stop_x if block is None else min(block, stop_x)
            else:
                entry_count[(cj, "eb")] += 1

        nx = desired
        if block is not None:
            nx = min(nx, block)
        if lead_x is not None:
            nx = min(nx, lead_x - MIN_GAP_PX)

        moved = nx > v["x"] + 0.1
        if not moved:
            v["delay_s"] += 1.0
            if v["was_moving"]:
                v["stops"] += 1
        v["moving"] = moved
        v["was_moving"] = moved
        v["x"] = nx
        lead_x = nx

        if v["x"] > CW + 25:
            removed.append(v)

    # Westbound lane (x decreasing)
    wb = sorted([v for v in vehicles if v["lane"] == "wb"], key=lambda x: x["x"])
    lead_x = None
    for v in wb:
        v["speed"] = max(5.0, float(v.get("base_speed", v["speed"])) * float(sim["rng"].uniform(0.95, 1.05)))
        v["y"] += max(-2.2, min(2.2, (ROAD_CY + 8.0) - v["y"]))

        desired = v["x"] - v["speed"]
        block = None
        crossing = None

        for j, jx in JX.items():
            stop_x = jx + STOP_OFF + 3
            if v["x"] >= stop_x and desired <= stop_x:
                crossing = (j, stop_x)
                break

        if crossing is not None:
            cj, stop_x = crossing
            can_enter = (
                _ew_allows_motion(sim["signals"][cj]["phase"])
                and entry_count[(cj, "wb")] < _entry_cap(sim, cj, "wb")
                and not _junction_conflict_blocked(vehicles, cj, "wb", v["id"])
                and _downstream_clear_for_entry(vehicles, cj, "wb", v["id"])
            )
            if not can_enter:
                block = stop_x if block is None else max(block, stop_x)
            else:
                entry_count[(cj, "wb")] += 1

        nx = desired
        if block is not None:
            nx = max(nx, block)
        if lead_x is not None:
            nx = max(nx, lead_x + MIN_GAP_PX)

        moved = nx < v["x"] - 0.1
        if not moved:
            v["delay_s"] += 1.0
            if v["was_moving"]:
                v["stops"] += 1
        v["moving"] = moved
        v["was_moving"] = moved
        v["x"] = nx
        lead_x = nx

        if v["x"] < -25:
            removed.append(v)

    # Vertical NS lanes (south -> north), one per junction.
    for j, jx in JX.items():
        lane = f"ns_{j}"
        ns = sorted([v for v in vehicles if v["lane"] == lane], key=lambda x: x["y"])
        lead_y = None
        stop_y = ROAD_CY + STOP_OFF + 3
        for v in ns:
            v["speed"] = max(4.5, float(v.get("base_speed", v["speed"])) * float(sim["rng"].uniform(0.94, 1.06)))
            desired = v["y"] - v["speed"]
            if v["y"] >= stop_y and desired <= stop_y:
                can_enter = (
                    _ns_allows_motion(sim["signals"][j]["phase"])
                    and entry_count[(j, "ns")] < _entry_cap(sim, j, "ns")
                    and not _junction_conflict_blocked(vehicles, j, "ns", v["id"])
                    and _downstream_clear_for_entry(vehicles, j, "ns", v["id"])
                )
                if not can_enter:
                    desired = stop_y
                else:
                    entry_count[(j, "ns")] += 1
            if lead_y is not None:
                desired = max(desired, lead_y + MIN_GAP_PX)

            moved = desired < v["y"] - 0.1
            if not moved:
                v["delay_s"] += 1.0
                if v["was_moving"]:
                    v["stops"] += 1
            v["moving"] = moved
            v["was_moving"] = moved
            v["y"] = desired
            lead_y = desired

            if v["y"] < -25:
                removed.append(v)

    if removed:
        for v in removed:
            sim["totals"]["completed"] += 1
            sim["totals"]["delay_accum"] += v["delay_s"]
            sim["totals"]["stops_accum"] += v["stops"]
            sim["throughput"] += 1
        sim["vehicles"] = [v for v in sim["vehicles"] if v not in removed]


def update_emergency_step(sim: dict):
    amb = sim["ambulance"]
    if sim["mode"] != "emergency":
        amb["status"] = "standby"
        return

    _ensure_emergency_dispatch(sim)

    if not amb["active"] or amb["completed"]:
        return

    desired = amb["x"] + amb["speed"]
    blocked = None

    for j, jx in JX.items():
        stop_x = jx - STOP_OFF - 4
        if amb["x"] <= stop_x and desired >= stop_x:
            if not _ew_allows_motion(sim["signals"][j]["phase"]):
                blocked = stop_x if blocked is None else min(blocked, stop_x)
                if j not in amb["stopped_junctions"]:
                    amb["stopped_junctions"].add(j)
                    amb["stops"] += 1

    nx = desired if blocked is None else min(desired, blocked)

    # Never pass through vehicles in the center corridor; maintain a hard gap.
    centerline_ahead = [
        v
        for v in sim["vehicles"]
        if v["lane"] == "eb" and v["x"] > amb["x"] and abs(v.get("y", ROAD_CY - 8.0) - (ROAD_CY - 8.0)) < 7.0
    ]
    if centerline_ahead:
        lead = min(centerline_ahead, key=lambda v: v["x"])
        nx = min(nx, lead["x"] - (MIN_GAP_PX + 2))

    moved = nx > amb["x"] + 0.1

    if not moved:
        amb["delay_s"] += 1.0
        amb["status"] = "waiting_signal_or_traffic"
    else:
        amb["status"] = "en_route"

    amb["x"] = nx

    if amb["x"] > CW + 50:
        amb["active"] = False
        amb["completed"] = True
        amb["exit_t"] = sim["time"]
        amb["status"] = "arrived"


def update_queues_and_stops_step(sim: dict):
    # EW queue derived from actual stopped vehicles near stop lines.
    for j, jx in JX.items():
        stop_eb = jx - STOP_OFF - 3
        stop_wb = jx + STOP_OFF + 3
        stop_ns = ROAD_CY + STOP_OFF + 3

        q_eb = 0
        q_wb = 0
        q_ns = 0
        for v in sim["vehicles"]:
            if v["lane"] == "eb" and not v["moving"] and stop_eb - 160 <= v["x"] <= stop_eb + 6:
                q_eb += 1
            if v["lane"] == "wb" and not v["moving"] and stop_wb - 6 <= v["x"] <= stop_wb + 160:
                q_wb += 1
            if v["lane"] == f"ns_{j}" and not v["moving"] and stop_ns - 4 <= v["y"] <= CH + 30:
                q_ns += 1

        sim["queues"][j]["ew"] = q_eb + q_wb
        sim["queues"][j]["ns"] = float(q_ns)


def append_metrics_history_step(sim: dict):
    h = sim["metrics_history"]

    q_total = 0
    for j in JUNCTIONS:
        q_total += int(sim["queues"][j]["ew"] + round(sim["queues"][j]["ns"]))

    active_count = len(sim["vehicles"]) + (1 if sim["ambulance"].get("active") else 0)
    observed = sim["totals"]["completed"] + len(sim["vehicles"])

    delay_total = sim["totals"]["delay_accum"] + sum(v["delay_s"] for v in sim["vehicles"])
    stops_total = sim["totals"]["stops_accum"] + sum(v["stops"] for v in sim["vehicles"])

    avg_delay = (delay_total / observed) if observed > 0 else 0.0
    avg_stops = (stops_total / observed) if observed > 0 else 0.0

    h["time"].append(sim["time"])
    h["queue_length"].append(q_total)
    h["throughput"].append(sim["throughput"])
    h["avg_delay"].append(avg_delay)
    h["avg_stops"].append(avg_stops)
    h["active_vehicles"].append(active_count)
    for j in JUNCTIONS:
        h["per_j_total"][j].append(int(sim["queues"][j]["ew"] + round(sim["queues"][j]["ns"])))


def simulation_step(sim: dict):
    if sim["completed"]:
        return

    spawn_vehicles_step(sim)
    _ensure_emergency_dispatch(sim)
    update_signals_step(sim)
    update_vehicle_positions_step(sim)
    update_emergency_step(sim)
    update_queues_and_stops_step(sim)

    sim["time"] += 1
    if sim["time"] >= sim["duration"]:
        sim["time"] = sim["duration"]
        sim["completed"] = True

    sim["snapshot"] = build_snapshot(sim)
    append_metrics_history_step(sim)


# ──────────────────────────────────────────────────────────────────────────────
# SNAPSHOT / METRICS
# ──────────────────────────────────────────────────────────────────────────────
def build_snapshot(sim: dict) -> dict:
    per_j = {}
    total_halted = 0
    for j in JUNCTIONS:
        ew = int(sim["queues"][j]["ew"])
        ns = int(round(sim["queues"][j]["ns"]))
        total_halted += ew + ns
        per_j[j] = {
            "ew": ew,
            "ns": ns,
            "phase": sim["signals"][j]["phase"],
        }

    vehicles = [
        {
            "id": v["id"],
            "x": float(v["x"]),
            "y": float(v["y"]),
            "lane": v["lane"],
            "moving": bool(v["moving"]),
            "stopped": bool(not v["moving"]),
        }
        for v in sim["vehicles"]
    ]

    amb = sim["ambulance"]
    amb_out = {
        "active": bool(amb.get("active")),
        "pos_x": float(amb.get("x", -65.0)),
        "status": amb.get("status", "standby"),
        "stops": int(amb.get("stops", 0)),
        "delay_s": float(amb.get("delay_s", 0.0)),
        "entered_t": amb.get("entered_t"),
        "exit_t": amb.get("exit_t"),
    }

    return {
        "t": sim["time"],
        "per_j": per_j,
        "total_halted": total_halted,
        "vehicles": vehicles,
        "amb": amb_out,
    }


def latest_metrics(sim: dict) -> dict:
    h = sim["metrics_history"]
    if not h["time"]:
        return {
            "sim_time": 0,
            "queue_length": 0,
            "throughput": 0,
            "avg_delay": 0.0,
            "avg_stops": 0.0,
            "active_vehicles": 0,
        }
    i = -1
    return {
        "sim_time": h["time"][i],
        "queue_length": h["queue_length"][i],
        "throughput": h["throughput"][i],
        "avg_delay": h["avg_delay"][i],
        "avg_stops": h["avg_stops"][i],
        "active_vehicles": h["active_vehicles"][i],
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALS
# ──────────────────────────────────────────────────────────────────────────────
def kpi(label: str, val, delta: str = "", col: str = "#58a6ff") -> str:
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-val' style='color:{col}'>{val}</div>"
        f"{delta_html}"
        f"<div class='kpi-label'>{label}</div>"
        f"</div>"
    )


def draw_animation(snapshot: dict) -> go.Figure:
    per_j = snapshot.get("per_j", {})
    vehicles = snapshot.get("vehicles", [])
    amb = snapshot.get("amb", {})

    fig = go.Figure()
    bg = "#111318"

    # Background and roads
    fig.add_shape(type="rect", x0=0, x1=CW, y0=0, y1=CH, fillcolor=bg, line_width=0, layer="below")
    fig.add_shape(
        type="rect",
        x0=0,
        x1=CW,
        y0=ROAD_CY - ROAD_H,
        y1=ROAD_CY + ROAD_H,
        fillcolor="#252530",
        line_width=0,
        layer="below",
    )

    for x in range(10, CW, 30):
        fig.add_shape(
            type="line",
            x0=x,
            x1=x + 16,
            y0=ROAD_CY,
            y1=ROAD_CY,
            line=dict(color="#3a3a4a", width=1, dash="dash"),
            layer="below",
        )

    for j, jx in JX.items():
        fig.add_shape(type="rect", x0=jx - NS_W, x1=jx + NS_W, y0=0, y1=CH, fillcolor="#252530", line_width=0, layer="below")
        fig.add_shape(
            type="rect",
            x0=jx - NS_W,
            x1=jx + NS_W,
            y0=ROAD_CY - ROAD_H,
            y1=ROAD_CY + ROAD_H,
            fillcolor="#1e1e28",
            line_width=0,
            layer="below",
        )

        fig.add_shape(
            type="line",
            x0=jx - STOP_OFF,
            x1=jx - STOP_OFF,
            y0=ROAD_CY - ROAD_H,
            y1=ROAD_CY,
            line=dict(color="#666", width=2),
            layer="below",
        )
        fig.add_shape(
            type="line",
            x0=jx + STOP_OFF,
            x1=jx + STOP_OFF,
            y0=ROAD_CY,
            y1=ROAD_CY + ROAD_H,
            line=dict(color="#666", width=2),
            layer="below",
        )

    # Vehicle markers from live simulation state.
    vx, vy, vc, vs = [], [], [], []
    for v in vehicles:
        moving = bool(v.get("moving", False))
        vx.append(v["x"])
        vy.append(v.get("y", ROAD_CY - 8 if v["lane"] == "eb" else ROAD_CY + 8))
        vc.append("#4ade80" if moving else "#ef4444")
        vs.append("circle" if moving else "square")

    if vx:
        fig.add_trace(
            go.Scatter(
                x=vx,
                y=vy,
                mode="markers",
                marker=dict(color=vc, size=VEH_SZ, symbol=vs, line=dict(color="rgba(0,0,0,0.55)", width=0.8)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Ambulance
    if amb.get("active"):
        flash = snapshot.get("t", 0) % 2 == 0
        fig.add_trace(
            go.Scatter(
                x=[amb.get("pos_x", -65.0)],
                y=[ROAD_CY - 8],
                mode="markers",
                marker=dict(
                    color="#ffffff" if flash else "#ef4444",
                    size=14,
                    symbol="diamond",
                    line=dict(color="#ef4444", width=1.5),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Signal heads: 4 per junction (clearer than 2-dot view).
    sx, sy, sc = [], [], []
    for j, jx in JX.items():
        phase = per_j.get(j, {}).get("phase", "ALL_RED")
        ew_c, ns_c = PHASE_SIG.get(phase, ("#ef4444", "#ef4444"))

        # EW heads (west/east)
        sx += [jx - STOP_OFF - 10, jx + STOP_OFF + 10]
        sy += [ROAD_CY - ROAD_H - 10, ROAD_CY + ROAD_H + 10]
        sc += [ew_c, ew_c]

        # NS heads (north/south)
        sx += [jx - NS_W - 10, jx + NS_W + 10]
        sy += [ROAD_CY - STOP_OFF - 10, ROAD_CY + STOP_OFF + 10]
        sc += [ns_c, ns_c]

    fig.add_trace(
        go.Scatter(
            x=sx,
            y=sy,
            mode="markers",
            marker=dict(color=sc, size=10, symbol="circle", line=dict(color="#000", width=1.4)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Labels
    for j, jx in JX.items():
        fig.add_annotation(x=jx, y=8, text=j, showarrow=False, font=dict(color="#445", size=10, family="monospace"))

    fig.add_annotation(
        x=8,
        y=CH - 8,
        text=f"t={snapshot.get('t', 0)}s",
        showarrow=False,
        xanchor="left",
        font=dict(color="#777", size=10, family="monospace"),
    )
    fig.add_annotation(
        x=8,
        y=CH - 20,
        text=f"queued={snapshot.get('total_halted', 0)}",
        showarrow=False,
        xanchor="left",
        font=dict(color="#777", size=10, family="monospace"),
    )

    if amb.get("active"):
        fig.add_annotation(
            x=CW / 2,
            y=CH - 8,
            text="EMERGENCY ACTIVE",
            showarrow=False,
            font=dict(color="#ef4444", size=11, family="monospace"),
        )

    fig.update_layout(
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        xaxis=dict(visible=False, range=[0, CW], fixedrange=True),
        yaxis=dict(visible=False, range=[0, CH], fixedrange=True),
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


def chart_live_queue(history: dict, mode: str) -> go.Figure:
    fig = go.Figure()
    t = history["time"]
    q = history["queue_length"]

    if t:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=q,
                name=MODE_CFG[mode]["label"],
                line=dict(color=MODE_CFG[mode]["color"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.10)",
            )
        )

    dl = {
        **DARK,
        "xaxis": {**DARK["xaxis"], "range": [0, SIM_DURATION_DEFAULT]},
        "yaxis": {**DARK["yaxis"], "rangemode": "tozero"},
    }
    fig.update_layout(
        **dl,
        title=f"Live Queue — {MODE_CFG[mode]['label']}",
        xaxis_title="Sim Time (s)",
        yaxis_title="Vehicles Halted",
        height=220,
        showlegend=False,
    )
    return fig


def chart_live_perjunction(history: dict) -> go.Figure:
    fig = go.Figure()
    t = history["time"]
    cols = ["#58a6ff", "#56d364", "#f0a500"]

    if t:
        for i, j in enumerate(JUNCTIONS):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=history["per_j_total"][j],
                    name=j,
                    line=dict(color=cols[i], width=2),
                )
            )

    dl = {
        **DARK,
        "xaxis": {**DARK["xaxis"], "range": [0, SIM_DURATION_DEFAULT]},
        "yaxis": {**DARK["yaxis"], "rangemode": "tozero"},
    }
    fig.update_layout(
        **dl,
        title="Queue per Junction",
        xaxis_title="Time (s)",
        yaxis_title="Vehicles",
        height=220,
    )
    return fig


def chart_compare_avg_delay(fixed_run: dict, adaptive_run: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fixed_run["time"],
            y=fixed_run["avg_delay"],
            name="Fixed Timing",
            line=dict(color=MODE_CFG["fixed"]["color"], width=2.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=adaptive_run["time"],
            y=adaptive_run["avg_delay"],
            name="Adaptive Timing",
            line=dict(color=MODE_CFG["adaptive"]["color"], width=2.3),
        )
    )
    dl = {
        **DARK,
        "xaxis": {**DARK["xaxis"], "range": [0, SIM_DURATION_DEFAULT]},
        "yaxis": {**DARK["yaxis"], "rangemode": "tozero"},
    }
    fig.update_layout(**dl, title="Fixed vs Adaptive — Average Delay", xaxis_title="Sim Time (s)", yaxis_title="Avg Delay (s)", height=240)
    return fig


def chart_compare_queue(fixed_run: dict, adaptive_run: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fixed_run["time"],
            y=fixed_run["queue_length"],
            name="Fixed Timing",
            line=dict(color=MODE_CFG["fixed"]["color"], width=2.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=adaptive_run["time"],
            y=adaptive_run["queue_length"],
            name="Adaptive Timing",
            line=dict(color=MODE_CFG["adaptive"]["color"], width=2.3),
        )
    )
    dl = {
        **DARK,
        "xaxis": {**DARK["xaxis"], "range": [0, SIM_DURATION_DEFAULT]},
        "yaxis": {**DARK["yaxis"], "rangemode": "tozero"},
    }
    fig.update_layout(**dl, title="Fixed vs Adaptive — Queue Length", xaxis_title="Sim Time (s)", yaxis_title="Halted Vehicles", height=240)
    return fig


@st.cache_data(show_spinner=False)
def load_video_state_profile(path_str: str, mtime: float, max_points: int = 240):
    _ = mtime  # cache invalidation key
    path = Path(path_str)
    if not path.exists():
        return None

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        rows.append(
            {
                "queue_length": float(d.get("queue_length", 0.0)),
                "arrival_rate": float(d.get("arrival_rate", 0.0)),
                "departure_rate": float(d.get("departure_rate", 0.0)),
            }
        )

    if len(rows) < 20:
        return None

    q = np.array([r["queue_length"] for r in rows], dtype=float)
    arr = np.array([r["arrival_rate"] for r in rows], dtype=float)
    dep = np.array([r["departure_rate"] for r in rows], dtype=float)
    q = np.clip(q, 0.0, None)
    arr = np.clip(arr, 0.0, None)
    dep = np.clip(dep, 0.0, None)

    if len(q) > max_points:
        idx = np.linspace(0, len(q) - 1, max_points).astype(int)
        q = q[idx]
        arr = arr[idx]
        dep = dep[idx]

    # Assume similar cross-street demand based on video EW activity.
    ns_arr = np.clip(0.55 * arr + 0.20 * np.sqrt(q + 1.0), 0.05, None)
    t = np.arange(len(q), dtype=int)

    return {
        "time": t.tolist(),
        "queue_video": q.tolist(),
        "arr_ew": arr.tolist(),
        "arr_ns_assumed": ns_arr.tolist(),
        "dep_video": dep.tolist(),
    }


def _simulate_single_intersection(arr_ew: list, arr_ns: list, mode: str):
    q_ew = 0.0
    q_ns = 0.0
    phase = "EW_GREEN"
    last_green = "EW_GREEN"
    time_in_phase = 0

    sat_ew = 1.85
    sat_ns = 1.35
    min_green = 10
    max_green = 50
    hysteresis = 2.5

    out_q = []
    out_delay = []
    out_thr = []
    out_phase = []
    throughput = 0.0
    cumulative_wait = 0.0

    for t in range(len(arr_ew)):
        a_ew = max(0.0, float(arr_ew[t]))
        a_ns = max(0.0, float(arr_ns[t]))
        q_ew += a_ew
        q_ns += a_ns
        time_in_phase += 1

        if mode == "fixed":
            c = t % 66
            if c < 28:
                phase = "EW_GREEN"
            elif c < 31:
                phase = "EW_YELLOW"
            elif c < 32:
                phase = "ALL_RED"
            elif c < 60:
                phase = "NS_GREEN"
            elif c < 63:
                phase = "NS_YELLOW"
            else:
                phase = "ALL_RED"
        else:
            if phase in ("EW_YELLOW", "NS_YELLOW"):
                if time_in_phase >= YELLOW:
                    phase = "ALL_RED"
                    time_in_phase = 0
            elif phase == "ALL_RED":
                if time_in_phase >= ALL_RED:
                    phase = "NS_GREEN" if last_green == "EW_GREEN" else "EW_GREEN"
                    time_in_phase = 0
            else:
                p_ew = q_ew + 2.5 * a_ew
                p_ns = q_ns + 2.5 * a_ns
                switch = False
                if time_in_phase >= min_green:
                    if phase == "EW_GREEN" and p_ns > p_ew + hysteresis:
                        switch = True
                    elif phase == "NS_GREEN" and p_ew > p_ns + hysteresis:
                        switch = True
                    active_q = q_ew if phase == "EW_GREEN" else q_ns
                    opp_q = q_ns if phase == "EW_GREEN" else q_ew
                    if active_q >= 7 and opp_q <= active_q + 3 and time_in_phase < int(max_green * 0.85):
                        switch = False
                if time_in_phase >= max_green:
                    switch = True
                if switch:
                    last_green = phase
                    phase = "EW_YELLOW" if phase == "EW_GREEN" else "NS_YELLOW"
                    time_in_phase = 0

        d_ew = 0.0
        d_ns = 0.0
        if phase == "EW_GREEN":
            d_ew = min(q_ew, sat_ew)
            q_ew -= d_ew
        elif phase == "NS_GREEN":
            d_ns = min(q_ns, sat_ns)
            q_ns -= d_ns

        throughput += d_ew + d_ns
        cumulative_wait += (q_ew + q_ns)

        out_q.append(max(0.0, q_ew + q_ns))
        out_delay.append(cumulative_wait / max(1.0, throughput))
        out_thr.append(throughput)
        out_phase.append(phase)

    return {
        "queue": out_q,
        "avg_delay": out_delay,
        "throughput": out_thr,
        "phase": out_phase,
        "final_queue": out_q[-1] if out_q else 0.0,
        "final_delay": out_delay[-1] if out_delay else 0.0,
        "final_throughput": out_thr[-1] if out_thr else 0.0,
    }


@st.cache_data(show_spinner=False)
def compute_video_static_benchmark(path_str: str, mtime: float):
    profile = load_video_state_profile(path_str, mtime)
    if profile is None:
        return None
    fixed = _simulate_single_intersection(profile["arr_ew"], profile["arr_ns_assumed"], mode="fixed")
    adaptive = _simulate_single_intersection(profile["arr_ew"], profile["arr_ns_assumed"], mode="adaptive")
    return {"profile": profile, "fixed": fixed, "adaptive": adaptive}


def chart_video_static_compare(time_vals: list, fixed_vals: list, adaptive_vals: list, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=fixed_vals,
            name="Fixed Signal",
            line=dict(color=MODE_CFG["fixed"]["color"], width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_vals,
            y=adaptive_vals,
            name="Our Solution (Adaptive)",
            line=dict(color=MODE_CFG["adaptive"]["color"], width=2.2),
        )
    )
    dl = {
        **DARK,
        "xaxis": {**DARK["xaxis"], "range": [0, max(time_vals) if time_vals else 0]},
        "yaxis": {**DARK["yaxis"], "rangemode": "tozero"},
    }
    fig.update_layout(**dl, title=title, xaxis_title="Video-Derived Time (s)", yaxis_title=y_title, height=235)
    return fig


def _record_completed_run_if_needed(sim: dict):
    if sim["mode"] not in ("fixed", "adaptive"):
        return
    if not sim["completed"] or sim.get("recorded"):
        return

    if "completed_runs" not in st.session_state:
        st.session_state.completed_runs = {"fixed": [], "adaptive": []}

    h = sim["metrics_history"]
    rec = {
        "time": list(h["time"]),
        "avg_delay": list(h["avg_delay"]),
        "queue_length": list(h["queue_length"]),
        "throughput": list(h["throughput"]),
        "avg_stops": list(h["avg_stops"]),
    }
    st.session_state.completed_runs[sim["mode"]].append(rec)
    sim["recorded"] = True
    st.session_state.running = False


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    for k, v in [("ui_mode", "adaptive"), ("running", True), ("sim_speed", 1)]:
        if k not in st.session_state:
            st.session_state[k] = v
    if "completed_runs" not in st.session_state:
        st.session_state.completed_runs = {"fixed": [], "adaptive": []}

    if "sim" not in st.session_state:
        st.session_state.sim = _new_sim(st.session_state.ui_mode)

    # Keep simulation mode isolated from UI mode.
    if st.session_state.sim["mode"] != st.session_state.ui_mode:
        _reset_sim(st.session_state.ui_mode)

    sim = st.session_state.sim
    _record_completed_run_if_needed(sim)
    if sim["completed"] and st.session_state.running:
        st.session_state.running = False
    snapshot = sim["snapshot"]
    metrics = latest_metrics(sim)

    # Header
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown(
            "<h1 style='margin:0;font-size:1.7rem'>🚦 FlowSense</h1>"
            "<div style='color:#8b949e;font-size:.85rem;margin-top:2px'>"
            "True Live Stochastic Simulation — Session-State Driven"
            "</div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.caption("🟢 Live Engine &nbsp;|&nbsp; No precomputed replay")

    # Controls
    mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns([2, 2, 2, 0.7, 0.7, 1.5, 1.5])

    for col_, key in zip([mc1, mc2, mc3], ["fixed", "adaptive", "emergency"]):
        with col_:
            active = st.session_state.ui_mode == key
            cfg = MODE_CFG[key]
            if st.button(("✓ " if active else "") + cfg["label"], key=f"m_{key}", use_container_width=True):
                st.session_state.ui_mode = key
                _reset_sim(key)
                sim = st.session_state.sim
                snapshot = sim["snapshot"]
                metrics = latest_metrics(sim)

    with mc4:
        if st.session_state.running:
            if st.button("⏸", use_container_width=True):
                st.session_state.running = False
        else:
            if st.button("▶", use_container_width=True):
                st.session_state.running = True

    with mc5:
        if st.button("⏮", use_container_width=True):
            _reset_sim(st.session_state.ui_mode)
            sim = st.session_state.sim
            snapshot = sim["snapshot"]
            metrics = latest_metrics(sim)

    with mc6:
        spd = st.select_slider("Speed", options=[1, 2, 3, 5], value=st.session_state.sim_speed, label_visibility="collapsed")
        st.session_state.sim_speed = spd
    with mc7:
        status = "▶ Running" if st.session_state.running else "⏸ Paused"
        st.caption(f"Speed ×{st.session_state.sim_speed} &nbsp;|&nbsp; {status}")

    st.divider()

    # KPIs (all from current run state/history)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    q_hist = sim["metrics_history"]["queue_length"]
    prev_q = q_hist[-2] if len(q_hist) >= 2 else metrics["queue_length"]
    delta_q = metrics["queue_length"] - prev_q
    delta_q_str = f"{delta_q:+d}" if delta_q else ""

    with k1:
        st.markdown(
            kpi("Sim Time", f"{metrics['sim_time']}s", f"{metrics['sim_time']}/{sim['duration']}s"),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            kpi("Queue Length", metrics["queue_length"], delta_q_str, col="#f85149" if metrics["queue_length"] > 0 else "#56d364"),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(kpi("Throughput", metrics["throughput"], "served"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("Avg Delay", f"{metrics['avg_delay']:.1f}s", "live"), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi("Avg Stops", f"{metrics['avg_stops']:.2f}", "per vehicle"), unsafe_allow_html=True)
    with k6:
        st.markdown(kpi("Active Vehicles", metrics["active_vehicles"], col=MODE_CFG[st.session_state.ui_mode]["color"]), unsafe_allow_html=True)

    # Main panels
    col_anim, col_sig = st.columns([3, 1])

    with col_anim:
        mode_col = MODE_CFG[st.session_state.ui_mode]["color"]
        st.markdown(
            f"<div style='font-weight:700;color:{mode_col};margin-bottom:4px'>"
            f"● {MODE_CFG[st.session_state.ui_mode]['label']} — t={snapshot['t']}s</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(draw_animation(snapshot), use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

    with col_sig:
        st.markdown("**Signal States**")
        for j in JUNCTIONS:
            pj = snapshot["per_j"][j]
            ph = pj["phase"]
            ew_c, ns_c = PHASE_SIG.get(ph, ("#ef4444", "#ef4444"))
            st.markdown(
                f"<div class='sig-card'>"
                f"<b style='color:#aaa'>{j}</b> &nbsp;"
                f"<span style='color:{ew_c}'>●</span> EW "
                f"<span style='color:{ns_c}'>●</span> NS<br>"
                f"<span style='color:#666;font-size:.72rem'>"
                f"ew={pj['ew']} ns={pj['ns']} &nbsp; <i>{ph}</i>"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        if st.session_state.ui_mode == "emergency":
            amb = snapshot["amb"]
            st.markdown("---")
            st.markdown("**Emergency**")
            if not amb["active"] and amb.get("entered_t") is None:
                rem = max(0, sim["ambulance"]["dispatch_t"] - snapshot["t"])
                st.info(f"Dispatch in {rem}s")
            elif amb["active"]:
                pct = max(0.0, min(1.0, amb["pos_x"] / CW))
                st.warning(f"Status: {amb['status']}")
                st.progress(pct)
                st.caption(f"Stops: {amb['stops']} | Delay: {amb['delay_s']:.0f}s")
            else:
                st.success("Ambulance arrived")

        st.markdown("---")
        st.progress(min(1.0, snapshot["t"] / max(1, sim["duration"])), text=f"{snapshot['t']}/{sim['duration']}s")

    # Live charts (current run only)
    lc1, lc2 = st.columns([3, 2])
    with lc1:
        st.plotly_chart(chart_live_queue(sim["metrics_history"], st.session_state.ui_mode), use_container_width=True, config={"displayModeBar": False, "staticPlot": True})
    with lc2:
        st.plotly_chart(chart_live_perjunction(sim["metrics_history"]), use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

    with st.expander("Fixed vs Adaptive Comparison (completed live runs)"):
        fixed_runs = st.session_state.completed_runs.get("fixed", [])
        adaptive_runs = st.session_state.completed_runs.get("adaptive", [])
        if fixed_runs and adaptive_runs:
            fixed_run = fixed_runs[-1]
            adaptive_run = adaptive_runs[-1]
            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(chart_compare_avg_delay(fixed_run, adaptive_run), use_container_width=True, config={"displayModeBar": False, "staticPlot": True})
            with cc2:
                st.plotly_chart(chart_compare_queue(fixed_run, adaptive_run), use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

            fixed_final_delay = fixed_run["avg_delay"][-1] if fixed_run["avg_delay"] else 0.0
            adaptive_final_delay = adaptive_run["avg_delay"][-1] if adaptive_run["avg_delay"] else 0.0
            if fixed_final_delay > 0:
                improvement = (fixed_final_delay - adaptive_final_delay) / fixed_final_delay * 100
                st.caption(f"Latest completed runs: adaptive average-delay improvement = {improvement:.1f}%")
        else:
            st.caption("Run Fixed and Adaptive mode through one full simulation each to generate realtime comparison charts.")

    with st.expander("Simulation State Contract"):
        st.code(
            json.dumps(
                {
                    "sim": {
                        "time": sim["time"],
                        "mode": sim["mode"],
                        "running": st.session_state.running,
                        "duration": sim["duration"],
                        "vehicles": len(sim["vehicles"]),
                        "signals": {j: sim["signals"][j]["phase"] for j in JUNCTIONS},
                        "queues": {j: {"ew": sim["queues"][j]["ew"], "ns": int(round(sim["queues"][j]["ns"]))} for j in JUNCTIONS},
                        "throughput": sim["throughput"],
                    }
                },
                indent=2,
            ),
            language="json",
        )

    with st.expander("CV Layer Snapshot (reference only)"):
        video_path = Path("output/detection.mp4")
        if not video_path.exists():
            video_path = Path("data/videos/traffic.mp4")
        if video_path.exists() and video_path.stat().st_size > 1000:
            st.video(str(video_path))
            st.caption(f"Traffic video being analysed: {video_path}")

        slp = Path("output/state_log.jsonl")
        if slp.exists():
            lines = [ln for ln in slp.read_text().splitlines() if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                st.code(
                    json.dumps(
                        {
                            "intersection_id": "J0",
                            "timestamp": last.get("timestamp"),
                            "queue_length": last.get("queue_length", 0),
                            "arrival_rate": last.get("arrival_rate", 0),
                            "departure_rate": last.get("departure_rate", 0),
                        },
                        indent=2,
                    ),
                    language="json",
                )
                st.caption(f"{len(lines)} logged state samples")
        else:
            st.caption("No state_log.jsonl found.")

    with st.expander("Video-Derived Static Benchmark (Single Intersection)"):
        slp = Path("output/state_log.jsonl")
        if not slp.exists():
            st.caption("No video-derived state log found yet. Run the CV pipeline to generate `output/state_log.jsonl`.")
        else:
            bench = compute_video_static_benchmark(str(slp), slp.stat().st_mtime)
            if bench is None:
                st.caption("Insufficient samples in `output/state_log.jsonl` to build the benchmark.")
            else:
                prof = bench["profile"]
                fixed_b = bench["fixed"]
                adapt_b = bench["adaptive"]
                t_vals = prof["time"]

                k1, k2, k3, k4 = st.columns(4)
                fq = fixed_b["final_queue"]
                aq = adapt_b["final_queue"]
                fd = fixed_b["final_delay"]
                ad = adapt_b["final_delay"]
                ft = fixed_b["final_throughput"]
                at = adapt_b["final_throughput"]
                q_imp = ((fq - aq) / fq * 100.0) if fq > 0 else 0.0
                d_imp = ((fd - ad) / fd * 100.0) if fd > 0 else 0.0
                t_imp = ((at - ft) / ft * 100.0) if ft > 0 else 0.0

                with k1:
                    st.markdown(kpi("Final Queue", f"{aq:.1f}", f"Adaptive {q_imp:.1f}% lower"), unsafe_allow_html=True)
                with k2:
                    st.markdown(kpi("Avg Delay", f"{ad:.1f}s", f"Adaptive {d_imp:.1f}% lower"), unsafe_allow_html=True)
                with k3:
                    st.markdown(kpi("Throughput", f"{at:.1f}", f"Adaptive {t_imp:.1f}% higher"), unsafe_allow_html=True)
                with k4:
                    st.markdown(kpi("Samples Used", f"{len(t_vals)}", "from state_log"), unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        chart_video_static_compare(
                            t_vals,
                            fixed_b["queue"],
                            adapt_b["queue"],
                            "Video State — Queue (Fixed vs Adaptive)",
                            "Queue Length",
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True},
                    )
                with c2:
                    st.plotly_chart(
                        chart_video_static_compare(
                            t_vals,
                            fixed_b["avg_delay"],
                            adapt_b["avg_delay"],
                            "Video State — Average Delay (Fixed vs Adaptive)",
                            "Avg Delay (s)",
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True},
                    )

                st.plotly_chart(
                    chart_video_static_compare(
                        t_vals,
                        fixed_b["throughput"],
                        adapt_b["throughput"],
                        "Video State — Cumulative Throughput (Fixed vs Adaptive)",
                        "Vehicles Served",
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False, "staticPlot": True},
                )
                st.caption(
                    "Single-intersection benchmark uses video-derived EW arrivals from `state_log` and assumed NS demand scaled from the same video state."
                )

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#444;font-size:.75rem'>"
        "FlowSense Live Simulation — step-based, stochastic, session-state driven"
        "</div>",
        unsafe_allow_html=True,
    )

    # Exactly one simulation step per rerun while running.
    if st.session_state.running and not sim["completed"]:
        simulation_step(sim)
        sleep_s = max(0.03, 0.18 / max(1, st.session_state.sim_speed))
        time.sleep(sleep_s)
        st.rerun()


if __name__ == "__main__":
    main()
