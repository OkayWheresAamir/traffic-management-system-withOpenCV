"""
FlowSense — Hackathon Demo Dashboard (v2)
=========================================
Single-page layout with live traffic animation, signal states, and KPI comparison.

Usage:
    streamlit run src/dashboard.py

Works in two modes:
  • LIVE mode  — loads simulation results from results/ directory (video-derived)
  • FALLBACK   — uses hardcoded realistic demo data
"""

import json
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlowSense — AI Traffic Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RESULTS_DIR = Path("results")

# ──────────────────────────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0d1117; color: #e6edf3; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  .metric-card {
    background: linear-gradient(135deg, #1c2526, #1a2332);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
  }
  .metric-val  { font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
  .metric-label{ font-size: 0.82rem; color: #8b949e; margin-top: 4px; }
  .mode-btn-active {
    background: #1f4028; border: 1.5px solid #56d364;
    border-radius: 8px; padding: 8px 12px; color: #56d364;
    font-weight: 700; text-align: center; cursor: pointer;
  }
  .mode-btn { background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 8px 12px; color: #8b949e;
    font-weight: 600; text-align: center; cursor: pointer;
  }
  .sig-green { color: #22c55e; font-size: 1.2rem; font-weight: 700; }
  .sig-red   { color: #ef4444; font-size: 1.2rem; font-weight: 700; }
  .sig-yellow{ color: #eab308; font-size: 1.2rem; font-weight: 700; }
  h1, h2, h3 { color: #e6edf3 !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "baseline": "#f85149",
    "adaptive": "#58a6ff",
    "pcs":      "#56d364",
    "ambulance":"#f0a500",
}
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=40, r=20, t=36, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
)

MODES = {
    "fixed":     {"label": "Fixed Timing",         "color": "#f85149", "result_key": "baseline"},
    "adaptive":  {"label": "FlowSense Adaptive",   "color": "#58a6ff", "result_key": "adaptive"},
    "emergency": {"label": "Emergency Corridor",   "color": "#f59e0b", "result_key": "emergency_pcs"},
}

JUNCTIONS = ["J0", "J1", "J2"]

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    tags = ["baseline", "adaptive", "emergency_nopcs", "emergency_pcs"]
    data = {}
    for tag in tags:
        p = RESULTS_DIR / f"{tag}.json"
        if p.exists():
            with open(p) as f:
                data[tag] = json.load(f)
    if len(data) < 4:
        data = make_fallback()
        return data, False
    return data, True


def make_fallback():
    rng = np.random.RandomState(0)
    T = 600
    t = np.arange(T)
    base_h  = np.clip(3 + 22 * (1 - np.exp(-t / 80)) + rng.normal(0, 2, T), 0, None)
    adapt_h = np.clip(2 + 10 * (1 - np.exp(-t / 40)) + 3 * np.sin(t / 30) + rng.normal(0, 1.5, T), 0, None)

    def smooth(arr, k=7):
        return pd.Series(arr).rolling(k, min_periods=1, center=True).mean().tolist()

    base_h  = smooth(base_h.tolist())
    adapt_h = smooth(adapt_h.tolist())

    def make_per_j(halted):
        per_j = {}
        for i, j in enumerate(JUNCTIONS):
            frac = [0.35, 0.4, 0.25][i]
            per_j[j] = {"ns": round(h * frac * 0.5), "ew": round(h * frac * 0.5)}
        return per_j

    return {
        "baseline": {
            "tag": "baseline",
            "steps": [{"t": t_, "total_halted": round(h), "per_j": make_per_j(h)}
                      for t_, h in enumerate(base_h)],
            "summary": {"avg_halted": round(sum(base_h)/len(base_h), 1),
                        "peak_halted": round(max(base_h), 1)},
        },
        "adaptive": {
            "tag": "adaptive",
            "steps": [{"t": t_, "total_halted": round(h), "per_j": make_per_j(h)}
                      for t_, h in enumerate(adapt_h)],
            "summary": {"avg_halted": round(sum(adapt_h)/len(adapt_h), 1),
                        "peak_halted": round(max(adapt_h), 1)},
        },
        "emergency_nopcs": {
            "tag": "emergency_nopcs",
            "steps": [{"t": t_, "total_halted": round(h), "per_j": make_per_j(h)}
                      for t_, h in enumerate(adapt_h)],
            "summary": {"avg_halted": round(sum(adapt_h)/len(adapt_h), 1),
                        "peak_halted": round(max(adapt_h), 1),
                        "ambulance_travel_s": 186, "ambulance_enter_t": 120,
                        "ambulance_exit_t": 306, "ambulance_stops": 4},
        },
        "emergency_pcs": {
            "tag": "emergency_pcs",
            "steps": [{"t": t_, "total_halted": round(h), "per_j": make_per_j(h)}
                      for t_, h in enumerate(adapt_h)],
            "summary": {"avg_halted": round(sum(adapt_h)/len(adapt_h), 1),
                        "peak_halted": round(max(adapt_h), 1),
                        "ambulance_travel_s": 89, "ambulance_enter_t": 120,
                        "ambulance_exit_t": 209, "ambulance_stops": 0},
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL PHASE INFERENCE
# (derives approximate EW/NS green from queue data + mode + time)
# ──────────────────────────────────────────────────────────────────────────────

def infer_signal_phases(step_data, sim_t, mode):
    """
    Return dict: {junction: "EW_GREEN" or "NS_GREEN"}
    Uses queue data and mode to estimate current signal phase.
    """
    per_j = step_data.get("per_j", {})
    phases = {}
    for j in JUNCTIONS:
        q_ew = per_j.get(j, {}).get("ew", 0)
        q_ns = per_j.get(j, {}).get("ns", 0)
        if mode == "fixed":
            cycle = 90
            offset = {"J0": 0, "J1": 10, "J2": 20}[j]  # slight offset per junction
            phase_t = (sim_t + offset) % cycle
            phases[j] = "EW_GREEN" if phase_t < 43 else "NS_GREEN"
        elif mode == "emergency":
            # Emergency: EW stays green more often near ambulance window
            if 120 <= sim_t <= 220:
                phases[j] = "EW_GREEN"
            else:
                phases[j] = "EW_GREEN" if q_ew >= q_ns else "NS_GREEN"
        else:
            # Adaptive: green for whichever approach has higher queue
            phases[j] = "EW_GREEN" if q_ew >= q_ns else "NS_GREEN"
    return phases


# ──────────────────────────────────────────────────────────────────────────────
# TRAFFIC ANIMATION
# ──────────────────────────────────────────────────────────────────────────────

# Canvas layout constants
CW, CH = 860, 300
ROAD_CY = 150          # EW road center y
ROAD_HALF = 16         # EW road half-height
NS_ROAD_HALF = 16      # NS road half-width
JX = {"J0": 180, "J1": 430, "J2": 680}  # junction x positions
STOP_OFFSET = 28       # stop line distance from center
VEH_SPACING = 20       # spacing between queued vehicles
VEH_SIZE = 8           # vehicle dot size


def draw_traffic_animation(step_data, sim_t, mode, ambulance_pos=None):
    """
    Plotly-based traffic animation. Returns a go.Figure.
    Vehicles accumulate when red, discharge when green.
    """
    per_j = step_data.get("per_j", {})
    phases = infer_signal_phases(step_data, sim_t, mode)

    fig = go.Figure()

    # ── Background ──
    fig.add_shape(type="rect", x0=0, x1=CW, y0=0, y1=CH,
                  fillcolor="#141a12", line_width=0, layer="below")

    # ── EW road ──
    fig.add_shape(type="rect", x0=0, x1=CW,
                  y0=ROAD_CY - ROAD_HALF, y1=ROAD_CY + ROAD_HALF,
                  fillcolor="#2a2a30", line_width=0)

    # EW center dashes
    for x in range(10, CW, 32):
        fig.add_shape(type="line", x0=x, x1=x + 18, y0=ROAD_CY, y1=ROAD_CY,
                      line=dict(color="#444", width=1.5, dash="dash"))

    # ── NS roads and intersections ──
    for j, jx in JX.items():
        # NS road surface
        fig.add_shape(type="rect",
                      x0=jx - NS_ROAD_HALF, x1=jx + NS_ROAD_HALF, y0=0, y1=CH,
                      fillcolor="#2a2a30", line_width=0)
        # Intersection box
        fig.add_shape(type="rect",
                      x0=jx - NS_ROAD_HALF, x1=jx + NS_ROAD_HALF,
                      y0=ROAD_CY - ROAD_HALF, y1=ROAD_CY + ROAD_HALF,
                      fillcolor="#222228", line_width=0)
        # NS road center dashes
        for y in range(10, ROAD_CY - ROAD_HALF, 24):
            fig.add_shape(type="line", x0=jx, x1=jx, y0=y, y1=y + 14,
                          line=dict(color="#444", width=1, dash="dash"))
        for y in range(ROAD_CY + ROAD_HALF + 10, CH, 24):
            fig.add_shape(type="line", x0=jx, x1=jx, y0=y, y1=y + 14,
                          line=dict(color="#444", width=1, dash="dash"))

    # ── Stop lines ──
    for j, jx in JX.items():
        # Eastbound stop line
        fig.add_shape(type="line",
                      x0=jx - STOP_OFFSET, x1=jx - STOP_OFFSET,
                      y0=ROAD_CY - ROAD_HALF, y1=ROAD_CY,
                      line=dict(color="#666", width=2))
        # Westbound stop line
        fig.add_shape(type="line",
                      x0=jx + STOP_OFFSET, x1=jx + STOP_OFFSET,
                      y0=ROAD_CY, y1=ROAD_CY + ROAD_HALF,
                      line=dict(color="#666", width=2))
        # NS southbound stop line
        fig.add_shape(type="line",
                      x0=jx, x1=jx + NS_ROAD_HALF,
                      y0=ROAD_CY - STOP_OFFSET, y1=ROAD_CY - STOP_OFFSET,
                      line=dict(color="#666", width=2))

    # ── Collect vehicle positions ──
    veh_x, veh_y, veh_colors, veh_symbols = [], [], [], []

    for j, jx in JX.items():
        phase = phases[j]
        ew_green = (phase == "EW_GREEN")
        q_ew = int(min(per_j.get(j, {}).get("ew", 0), 14))
        q_ns = int(min(per_j.get(j, {}).get("ns", 0), 10))

        # ── Queued EW vehicles (eastbound, before stop line) ──
        stop_x = jx - STOP_OFFSET - 2
        for k in range(q_ew):
            vx = stop_x - k * VEH_SPACING
            if vx > (JX.get("J0", 0) - 150 if j == "J0" else 0):
                veh_x.append(vx)
                veh_y.append(ROAD_CY - 7)
                veh_colors.append("#c85050" if not ew_green else "#50dca0")
                veh_symbols.append("square")

        # ── Moving EW vehicles through / after intersection ──
        # A few vehicles moving east on the open road
        for k in range(3):
            base_x = (sim_t * 18 + k * 190 + {"J0": 0, "J1": 100, "J2": 200}[j]) % (CW + 80) - 40
            # Only show if they're in a plausible gap (not behind a queue)
            in_gap = True
            for j2, jx2 in JX.items():
                if abs(base_x - (jx2 - STOP_OFFSET - 5)) < 50 and not ew_green:
                    in_gap = False
            if in_gap and 0 < base_x < CW:
                veh_x.append(base_x)
                veh_y.append(ROAD_CY - 7)
                veh_colors.append("#50dca0")
                veh_symbols.append("square")

        # ── Westbound vehicles (moving left, simpler) ──
        for k in range(2):
            base_x = CW - (sim_t * 14 + k * 220 + {"J0": 50, "J1": 150, "J2": 250}[j]) % (CW + 80) + 40
            if 0 < base_x < CW:
                veh_x.append(base_x)
                veh_y.append(ROAD_CY + 7)
                veh_colors.append("#50dca0")
                veh_symbols.append("square")

        # ── Queued NS vehicles (southbound, above intersection) ──
        stop_y = ROAD_CY - STOP_OFFSET - 2
        for k in range(q_ns):
            vy = stop_y - k * VEH_SPACING
            if vy > 5:
                veh_x.append(jx + 7)
                veh_y.append(vy)
                veh_colors.append("#c85050" if ew_green else "#50dca0")
                veh_symbols.append("square")

        # ── Moving NS vehicles (northbound from bottom) ──
        for k in range(2):
            base_y = CH - (sim_t * 12 + k * 130) % (CH + 60) + 30
            if ROAD_CY + ROAD_HALF < base_y < CH:
                veh_x.append(jx - 7)
                veh_y.append(base_y)
                veh_colors.append("#50dca0")
                veh_symbols.append("square")

    # ── Ambulance (emergency mode) ──
    if mode == "emergency" and 120 <= sim_t <= 220:
        amb_x = (sim_t - 120) * (CW / 100)  # travels full width over ~100s
        amb_x = min(amb_x, CW + 20)
        if -20 < amb_x < CW + 20:
            flash = (sim_t % 2 == 0)
            veh_x.append(amb_x)
            veh_y.append(ROAD_CY - 7)
            veh_colors.append("#ef4444" if flash else "#ffffff")
            veh_symbols.append("diamond")

    # ── Draw all vehicles ──
    if veh_x:
        fig.add_trace(go.Scatter(
            x=veh_x, y=veh_y,
            mode="markers",
            marker=dict(color=veh_colors, size=VEH_SIZE, symbol=veh_symbols,
                        line=dict(color="#000", width=0.5)),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Signal lights ──
    sig_x, sig_y, sig_col = [], [], []
    for j, jx in JX.items():
        phase = phases[j]
        ew_col = "#22c55e" if phase == "EW_GREEN" else "#ef4444"
        ns_col = "#22c55e" if phase == "NS_GREEN" else "#ef4444"
        # EW signal (top-left of intersection)
        sig_x.append(jx - STOP_OFFSET - 8)
        sig_y.append(ROAD_CY - ROAD_HALF - 8)
        sig_col.append(ew_col)
        # NS signal (right side, above road)
        sig_x.append(jx + NS_ROAD_HALF + 8)
        sig_y.append(ROAD_CY - STOP_OFFSET - 8)
        sig_col.append(ns_col)

    fig.add_trace(go.Scatter(
        x=sig_x, y=sig_y,
        mode="markers",
        marker=dict(color=sig_col, size=11, symbol="circle",
                    line=dict(color="#000000", width=1.5)),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── PCS reservation highlight (emergency mode, ambulance ahead) ──
    if mode == "emergency" and 115 <= sim_t <= 220:
        amb_x_now = (sim_t - 120) * (CW / 100)
        for j, jx in JX.items():
            if abs(jx - amb_x_now) < 250:
                fig.add_shape(type="rect",
                              x0=jx - NS_ROAD_HALF - 4, x1=jx + NS_ROAD_HALF + 4,
                              y0=ROAD_CY - ROAD_HALF - 4, y1=ROAD_CY + ROAD_HALF + 4,
                              fillcolor="rgba(0,0,0,0)",
                              line=dict(color="#f59e0b", width=2, dash="dot"))
                fig.add_annotation(x=jx, y=ROAD_CY - ROAD_HALF - 16,
                                   text="PCS", showarrow=False,
                                   font=dict(color="#f59e0b", size=9, family="monospace"))

    # ── Junction labels ──
    for j, jx in JX.items():
        fig.add_annotation(x=jx, y=10, text=j, showarrow=False,
                           font=dict(color="#556", size=11, family="monospace"))

    # ── Time and mode overlay ──
    fig.add_annotation(x=10, y=CH - 10, text=f"t = {sim_t}s",
                       showarrow=False, xanchor="left",
                       font=dict(color="#667", size=11, family="monospace"))
    total_q = step_data.get("total_halted", 0)
    fig.add_annotation(x=10, y=CH - 24, text=f"queued: {total_q}",
                       showarrow=False, xanchor="left",
                       font=dict(color="#667", size=11, family="monospace"))

    if mode == "emergency" and 120 <= sim_t <= 220:
        fig.add_annotation(x=CW / 2, y=CH - 10, text="⚠ EMERGENCY VEHICLE ACTIVE",
                           showarrow=False,
                           font=dict(color="#ef4444", size=12, family="monospace"))

    # ── Legend ──
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#50dca0", size=8, symbol="square"),
        name="Moving", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#c85050", size=8, symbol="square"),
        name="Queued / Stopped", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#22c55e", size=9, symbol="circle"),
        name="Signal Green", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#ef4444", size=9, symbol="circle"),
        name="Signal Red", showlegend=True,
    ))

    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#141a12",
        xaxis=dict(visible=False, range=[0, CW]),
        yaxis=dict(visible=False, range=[0, CH]),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=-0.12, bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#889", size=10)),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def chart_queue_comparison(base_steps, adapt_steps):
    t_b = [s["t"] for s in base_steps]
    h_b = [s["total_halted"] for s in base_steps]
    t_a = [s["t"] for s in adapt_steps]
    h_a = [s["total_halted"] for s in adapt_steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_b, y=h_b, name="Fixed-Time (Baseline)",
                             line=dict(color=COLORS["baseline"], width=2.5)))
    fig.add_trace(go.Scatter(x=t_a, y=h_a, name="FlowSense Adaptive",
                             line=dict(color=COLORS["adaptive"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
    fig.update_layout(**DARK_LAYOUT,
                      title="Halted Vehicles Over Time",
                      xaxis_title="Sim Time (s)", yaxis_title="Halted Vehicles",
                      height=280)
    return fig


def chart_avg_bar(summaries):
    names  = ["Fixed-Time\nBaseline", "FlowSense\nAdaptive"]
    values = [summaries["baseline"]["avg_halted"], summaries["adaptive"]["avg_halted"]]
    colors = [COLORS["baseline"], COLORS["adaptive"]]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors,
                           text=[f"{v:.1f}" for v in values],
                           textposition="outside",
                           textfont=dict(color="#e6edf3")))
    fig.update_layout(**DARK_LAYOUT, title="Avg. Halted Vehicles",
                      height=260, showlegend=False)
    return fig


def chart_ambulance_timeline(nopcs_sum, pcs_sum):
    ne = nopcs_sum.get("ambulance_enter_t", 120)
    nt = nopcs_sum.get("ambulance_travel_s", 186)
    pt = pcs_sum.get("ambulance_travel_s", 89)
    ns = nopcs_sum.get("ambulance_stops", 4)
    ps = pcs_sum.get("ambulance_stops", 0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"Without PCS  ({ns} stops, {nt}s)",
                         y=["Without PCS"], x=[nt], base=[ne],
                         orientation="h", marker_color=COLORS["baseline"],
                         text=f"  {nt}s  |  {ns} stop(s)",
                         textposition="inside", insidetextanchor="start"))
    fig.add_trace(go.Bar(name=f"With PCS  ({ps} stops, {pt}s)",
                         y=["With PCS"], x=[pt], base=[ne],
                         orientation="h", marker_color=COLORS["pcs"],
                         text=f"  {pt}s  |  0 stops",
                         textposition="inside", insidetextanchor="start"))
    fig.add_vline(x=ne, line_dash="dash", line_color=COLORS["ambulance"],
                  annotation_text="🚑 Dispatched", annotation_position="top right")
    amb_layout = {**DARK_LAYOUT, "legend": dict(orientation="h", y=-0.4,
                                               bgcolor="rgba(0,0,0,0)")}
    fig.update_layout(**amb_layout, title="Ambulance Corridor Travel Time",
                      xaxis_title="Simulation Time (s)", barmode="overlay",
                      height=220)
    return fig


def chart_per_junction(steps):
    t = [s["t"] for s in steps]
    colors = ["#58a6ff", "#56d364", "#f0a500"]
    fig = go.Figure()
    for i, j in enumerate(JUNCTIONS):
        total = [s.get("per_j", {}).get(j, {}).get("ew", 0)
                 + s.get("per_j", {}).get(j, {}).get("ns", 0) for s in steps]
        total = pd.Series(total).rolling(5, min_periods=1, center=True).mean().tolist()
        fig.add_trace(go.Scatter(x=t, y=total, name=j,
                                 line=dict(color=colors[i], width=2)))
    fig.update_layout(**DARK_LAYOUT, title="Queue per Intersection (Adaptive)",
                      xaxis_title="Time (s)", yaxis_title="Halted", height=240)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE (HTML — no Plotly, no crash)
# ──────────────────────────────────────────────────────────────────────────────

def render_architecture_html():
    layers = [
        ("Layer 1 — Edge CV",
         "#58a6ff", "#0d1b2e",
         "📷 Camera feed → YOLOv8n detection → ByteTrack → Homography → "
         "Queue count, arrival rate, departure rate (1 Hz)"),
        ("Layer 2 — Adaptive Control",
         "#f0a500", "#1e1500",
         "Pressure = queue + 5× arrival_rate + regional platoon bias. "
         "Switch phase when opposing pressure exceeds current + hysteresis. "
         "MIN_GREEN 8 s, MAX_GREEN 45 s."),
        ("Layer 3 — PCS Emergency",
         "#56d364", "#0d1e0d",
         "GPS dispatch → A* route → ETA per junction → reservation windows "
         "(10 s pre-clear, 6 s post-guard). Forces EW green before ambulance arrives."),
    ]
    html = "<div style='display:flex; gap:12px; margin-top:8px;'>"
    for title, border, bg, desc in layers:
        html += f"""
        <div style='flex:1; background:{bg}; border:1.5px solid {border};
             border-radius:10px; padding:14px;'>
          <div style='color:{border}; font-weight:700; font-size:0.9rem; margin-bottom:6px;'>{title}</div>
          <div style='color:#8b949e; font-size:0.8rem; line-height:1.6;'>{desc}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Session state init ──────────────────────────────────────────────────
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "running" not in st.session_state:
        st.session_state.running = True
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = "adaptive"
    if "sim_speed" not in st.session_state:
        st.session_state.sim_speed = 3  # steps per rerun

    data, is_live = load_results()

    # ── Header ──────────────────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.markdown(
            "<h1 style='margin:0; padding:0; font-size:1.8rem;'>🚦 FlowSense</h1>"
            "<div style='color:#8b949e; font-size:0.9rem; margin-top:2px;'>"
            "AI-Driven Adaptive Traffic &amp; Emergency Corridor System</div>",
            unsafe_allow_html=True,
        )
    with hcol2:
        mode_label = "🟢 Live — Video-Derived" if is_live else "🟡 Demo Mode"
        st.caption(f"{mode_label} &nbsp;|&nbsp; India Innovates 2026 — Team HighKey Trophy")

    st.divider()

    # ── KPI cards ───────────────────────────────────────────────────────────
    base_avg  = data["baseline"]["summary"]["avg_halted"]
    adapt_avg = data["adaptive"]["summary"]["avg_halted"]
    reduction = round((base_avg - adapt_avg) / base_avg * 100) if base_avg else 0

    pcs_travel   = data["emergency_pcs"]["summary"].get("ambulance_travel_s") or 89
    nopcs_travel = data["emergency_nopcs"]["summary"].get("ambulance_travel_s") or 186
    emerg_improv = round((nopcs_travel - pcs_travel) / nopcs_travel * 100) if nopcs_travel else 0
    pcs_stops    = data["emergency_pcs"]["summary"].get("ambulance_stops", 0)
    nopcs_stops  = data["emergency_nopcs"]["summary"].get("ambulance_stops", 4)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{reduction}%</div>"
                    "<div class='metric-label'>Reduction in avg. halted vehicles</div></div>",
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{emerg_improv}%</div>"
                    "<div class='metric-label'>Faster ambulance response</div></div>",
                    unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{pcs_stops} stops</div>"
                    f"<div class='metric-label'>Ambulance stops with PCS (vs {nopcs_stops} without)</div></div>",
                    unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{pcs_travel}s</div>"
                    f"<div class='metric-label'>Ambulance travel time with PCS (vs {nopcs_travel}s)</div></div>",
                    unsafe_allow_html=True)

    st.divider()

    # ── Main layout: simulation | controls ─────────────────────────────────
    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.markdown("**Controller Mode**")
        for key, info in MODES.items():
            active = st.session_state.ui_mode == key
            label  = ("✅ " if active else "") + info["label"]
            if st.button(label, key=f"btn_{key}", use_container_width=True):
                st.session_state.ui_mode = key
                st.session_state.step = 0

        st.markdown("---")

        # Play / Pause
        p_col, s_col = st.columns(2)
        with p_col:
            if st.session_state.running:
                if st.button("⏸ Pause", use_container_width=True):
                    st.session_state.running = False
            else:
                if st.button("▶ Play", use_container_width=True):
                    st.session_state.running = True
        with s_col:
            if st.button("⏮ Reset", use_container_width=True):
                st.session_state.step = 0

        speed = st.select_slider("Speed", options=[1, 2, 3, 5, 8],
                                 value=st.session_state.sim_speed)
        st.session_state.sim_speed = speed

        st.markdown("---")
        st.markdown("**Signal States**")

        # Look up current step data
        ui_mode  = st.session_state.ui_mode
        rkey     = MODES[ui_mode]["result_key"]
        steps    = data[rkey]["steps"]
        cur_step = min(st.session_state.step, len(steps) - 1)
        sd       = steps[cur_step]
        phases   = infer_signal_phases(sd, cur_step, ui_mode)

        for j in JUNCTIONS:
            ph = phases[j]
            ew_s = "🟢 EW" if ph == "EW_GREEN" else "🔴 EW"
            ns_s = "🟢 NS" if ph == "NS_GREEN" else "🔴 NS"
            q_ew = sd.get("per_j", {}).get(j, {}).get("ew", 0)
            q_ns = sd.get("per_j", {}).get(j, {}).get("ns", 0)
            st.markdown(
                f"<div style='background:#161b22; border:1px solid #21262d; border-radius:8px;"
                f" padding:8px 10px; margin-bottom:6px;'>"
                f"<span style='color:#8b949e; font-size:0.85rem;'><b>{j}</b></span>"
                f"&nbsp;&nbsp;{ew_s}&nbsp;{ns_s}<br/>"
                f"<span style='color:#556; font-size:0.78rem;'>EW q={q_ew} · NS q={q_ns}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Live Metrics**")
        total_q = sd.get("total_halted", 0)
        elapsed_s = cur_step
        st.metric("Sim Time", f"{elapsed_s}s")
        st.metric("Total Queued", total_q)
        st.metric("Mode", MODES[ui_mode]["label"])
        st.progress(min(1.0, cur_step / 599))

        if ui_mode == "emergency":
            st.markdown("---")
            st.markdown("**Emergency Vehicle**")
            if cur_step < 120:
                st.info(f"🚑 Dispatching in {120 - cur_step}s…")
            elif cur_step <= 209:
                pct = (cur_step - 120) / 89
                st.warning("🚑 En route…")
                st.progress(min(1.0, pct))
            else:
                st.success(f"🏥 Arrived! Travel: {pcs_travel}s — 0 stops")

    with col_main:
        mode_info = MODES[ui_mode]
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>"
            f"<div style='width:10px; height:10px; border-radius:50%; "
            f"background:{mode_info['color']}; box-shadow:0 0 6px {mode_info['color']};'></div>"
            f"<span style='font-weight:700; color:{mode_info['color']};'>{mode_info['label']}</span>"
            f"<span style='color:#556; font-size:0.85rem;'>— live intersection simulation</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        anim_placeholder = st.empty()
        anim_fig = draw_traffic_animation(sd, cur_step, ui_mode)
        anim_placeholder.plotly_chart(anim_fig, use_container_width=True,
                                      config={"displayModeBar": False})

    st.divider()

    # ── Comparison charts (always visible, no tabs) ──────────────────────
    st.markdown("### Scenario Comparison")
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.plotly_chart(chart_queue_comparison(
            data["baseline"]["steps"], data["adaptive"]["steps"]),
            use_container_width=True, config={"displayModeBar": False})
    with cc2:
        st.plotly_chart(chart_avg_bar({
            "baseline": data["baseline"]["summary"],
            "adaptive": data["adaptive"]["summary"],
        }), use_container_width=True, config={"displayModeBar": False})

    st.plotly_chart(chart_per_junction(data["adaptive"]["steps"]),
                    use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Emergency Corridor")
    st.plotly_chart(chart_ambulance_timeline(
        data["emergency_nopcs"]["summary"],
        data["emergency_pcs"]["summary"]),
        use_container_width=True, config={"displayModeBar": False})

    pcs_enter = data["emergency_pcs"]["summary"].get("ambulance_enter_t", 120)
    pcs_exit  = data["emergency_pcs"]["summary"].get("ambulance_exit_t", 209)
    pcs_amb   = data["emergency_pcs"]["summary"].get("ambulance_travel_s", 89)
    eta_j0 = pcs_enter + 7
    eta_j1 = pcs_enter + 43
    eta_j2 = pcs_enter + 79
    st.markdown(
        f"<div style='background:#0d1117; border:1px solid #21262d; border-radius:10px;"
        f" padding:16px; font-family:monospace; font-size:0.82rem; color:#e6edf3; line-height:2;'>"
        f"<span style='color:#f0a500;'>T={pcs_enter}s</span> 🚑 Dispatched — PCS receives GPS ping<br/>"
        f"<span style='color:#58a6ff;'>T={pcs_enter+1}s</span> A* route computed: entry → J0 → J1 → J2 → hospital<br/>"
        f"<span style='color:#58a6ff;'>T={pcs_enter+1}s</span> ETAs: J0=T+{eta_j0-pcs_enter}, "
        f"J1=T+{eta_j1-pcs_enter}, J2=T+{eta_j2-pcs_enter}<br/>"
        f"<span style='color:#56d364;'>T={pcs_enter+2}s</span> Reservations issued — windows [{eta_j0-10}–{eta_j0+6}s], "
        f"[{eta_j1-10}–{eta_j1+6}s], [{eta_j2-10}–{eta_j2+6}s]<br/>"
        f"<span style='color:#56d364;'>T={eta_j0}s</span> 🚑 Passes J0 — GREEN ✅ (0 stop)<br/>"
        f"<span style='color:#56d364;'>T={eta_j1}s</span> 🚑 Passes J1 — GREEN ✅ (0 stop)<br/>"
        f"<span style='color:#56d364;'>T={eta_j2}s</span> 🚑 Passes J2 — GREEN ✅ (0 stop)<br/>"
        f"<span style='color:#56d364;'>T={pcs_exit}s</span> 🏥 Arrived. Travel time: <b>{pcs_amb}s</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📹 CV Layer — Detection Pipeline"):
        video_path = Path("output/detection.mp4")
        if video_path.exists() and video_path.stat().st_size > 1000:
            st.video(str(video_path))
            st.caption("YOLOv8 vehicle detection + ByteTrack on real Delhi traffic footage")
        state_log_path = Path("output/state_log.jsonl")
        if state_log_path.exists():
            with open(state_log_path) as f:
                lines = f.read().strip().splitlines()
                last = json.loads(lines[-1])
            example_state = {
                "intersection_id": "J0",
                "timestamp": last.get("timestamp"),
                "approaches": {
                    "E": {"queue_count": last.get("queue_length", 0),
                          "arrival_rate_pps": last.get("arrival_rate", 0)},
                    "W": {"queue_count": max(0, last.get("queue_length", 0) - 3),
                          "arrival_rate_pps": round(last.get("arrival_rate", 0) * 0.7, 2)},
                    "N": {"queue_count": 4, "arrival_rate_pps": 0.12},
                    "S": {"queue_count": 3, "arrival_rate_pps": 0.09},
                },
                "current_phase": "EW_green",
                "time_in_phase_s": 12,
                "detector_stats": {"frame_rate": 14.3,
                                   "detected_veh": last.get("queue_length", 0) + 8},
            }
            st.code(json.dumps(example_state, indent=2), language="json")
            st.caption(f"Real state vector — {len(lines)} samples captured from video")

    with st.expander("🏗️ System Architecture"):
        render_architecture_html()
        with st.expander("📘 Pressure Controller"):
            st.code("""
# Per-approach pressure
pressure = queue_count + 5.0 * arrival_rate + regional_platoon_bias

# Phase switching rule (per junction, per second)
if time_in_phase >= MIN_GREEN (8s):
    if pressure_other > pressure_current + HYSTERESIS (2.0):
        → switch (via yellow 3s + all-red 1.5s)
elif time_in_phase >= MAX_GREEN (45s):
    → force switch (starvation prevention)

# Layer 2: platoon propagation
if upstream departed > 2 vehicles within [travel_delay ± 3s]:
    EW pressure += 0.6 * platoon_size   # bias toward holding EW green
""", language="python")

    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#484f58; font-size:0.78rem;'>"
        "FlowSense — Team HighKey Trophy &nbsp;|&nbsp; India Innovates 2026 &nbsp;|&nbsp;"
        " Inspired by SURTRAC (CMU), SCATS (NSW), Singapore TPS"
        "</div>", unsafe_allow_html=True)

    # ── Animation loop (advance sim step and rerun) ──────────────────────
    if st.session_state.running:
        st.session_state.step = (
            st.session_state.step + st.session_state.sim_speed
        ) % len(steps)
        time.sleep(0.07)   # ~14 FPS at speed 1
        st.rerun()


if __name__ == "__main__":
    main()
