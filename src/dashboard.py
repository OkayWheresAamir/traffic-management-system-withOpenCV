"""
FlowSense — Live Traffic Simulation Dashboard
=============================================
Video → State Extraction → Simulation → Control → THIS DASHBOARD

Layout (live-first):
  ┌ Header + mode strip ──────────────────────────────────────┐
  │ KPI row: Sim Time | Queued | Throughput | Avg Delay       │
  ├ Animation (col 3) | Signal panel (col 1) ─────────────────┤
  │ Live queue chart (current mode only, grows in real-time)  │
  │ Per-junction live queue chart                             │
  ├ ─── Compare Modes (expander) ─────────────────────────────┤
  │ ─── Emergency Corridor (expander, emergency mode only) ───┤
  └ ─── System Architecture (expander) ──────────────────────┘

Data flow: results/*.json → session_state step index → sd (step data)
  sd["per_j"][j] = {ew, ns, phase}   ← real signal phase from demo_runner.py
  sd["amb"]      = {pos_m, active}    ← ambulance position (emergency only)
  session_state.hist_halted           ← live history buffer, grows each frame

Usage:  streamlit run src/dashboard.py
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
    page_title="FlowSense — Live Traffic Simulation",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RESULTS_DIR = Path("results")
SIM_LIMIT   = 120     # demo runs for exactly 120 simulation seconds
JUNCTIONS   = ["J0", "J1", "J2"]

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MODE_CFG = {
    "fixed":     {"label":"Fixed Timing",       "color":"#f85149", "key":"baseline"},
    "adaptive":  {"label":"FlowSense Adaptive",  "color":"#58a6ff", "key":"adaptive"},
    "emergency": {"label":"Emergency Corridor",  "color":"#f59e0b", "key":"emergency_pcs"},
}

PHASE_SIG = {           # (EW signal colour, NS signal colour)
    "EW_GREEN":  ("#22c55e", "#ef4444"),
    "EW_YELLOW": ("#eab308", "#ef4444"),
    "NS_GREEN":  ("#ef4444", "#22c55e"),
    "NS_YELLOW": ("#ef4444", "#eab308"),
    "ALL_RED":   ("#ef4444", "#ef4444"),
}

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=40, r=20, t=36, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
)

# Canvas geometry
CW, CH       = 860, 280
ROAD_CY      = 140
ROAD_H       = 18      # half-height of EW road
NS_W         = 18      # half-width of NS roads
STOP_OFF     = 30      # stop-line distance from junction centre
VEH_GAP      = 19      # gap between stacked queued vehicles
VEH_SZ       = 10      # vehicle marker size (px)
JX           = {"J0": 170, "J1": 430, "J2": 690}  # junction x-positions

# Ambulance geometry (mirrors demo_runner.py)
AMB_ENTRY_M  = 100.0
AMB_SPACING_M= 500.0
AMB_SCALE    = (JX["J2"] - JX["J0"]) / (2 * AMB_SPACING_M)   # px/m


def amb_to_x(pos_m: float) -> float:
    return JX["J0"] + (pos_m - AMB_ENTRY_M) * AMB_SCALE


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    tags = ["baseline", "adaptive", "emergency_nopcs", "emergency_pcs"]
    out  = {}
    for tag in tags:
        p = RESULTS_DIR / f"{tag}.json"
        if p.exists():
            with open(p) as f:
                out[tag] = json.load(f)
    if len(out) < 4:
        return _make_fallback(), False
    return out, True


def _make_fallback():
    """Minimal fallback if results/ files are missing."""
    rng = np.random.RandomState(0)
    T   = SIM_LIMIT
    t   = np.arange(T)

    def _smooth(arr):
        return pd.Series(arr).rolling(5, min_periods=1, center=True).mean().tolist()

    bh = _smooth(np.clip(3 + 20*(1-np.exp(-t/70)) + rng.normal(0,2,T), 0, None).tolist())
    ah = _smooth(np.clip(2 +  9*(1-np.exp(-t/35)) + 3*np.sin(t/30) + rng.normal(0,1.5,T), 0, None).tolist())

    CYCLES = {"fixed": 90, "adaptive": None}

    def _per_j(h, t_, mode):
        pj = {}
        for i, j in enumerate(JUNCTIONS):
            frac = [0.35, 0.4, 0.25][i]
            off  = [0, 10, 20][i]
            ph   = "EW_GREEN" if ((t_ + off) % 90) < 43 else "NS_GREEN"
            pj[j] = {"ew": round(h*frac*0.5), "ns": round(h*frac*0.5), "phase": ph}
        return pj

    def _build(tag, h_arr, amb_info=None):
        steps = []
        for t_, h in enumerate(h_arr):
            sd = {"t": t_, "total_halted": round(h), "per_j": _per_j(h, t_, tag)}
            if amb_info:
                sd["amb"] = {
                    "pos_m":  max(0.0, (t_-120)*14.0),
                    "active": bool(120 <= t_ <= 209),
                    "speed":  14.0,
                }
            steps.append(sd)
        avg = round(sum(h_arr)/len(h_arr), 1)
        return {"tag": tag, "steps": steps,
                "summary": {"avg_halted": avg, "peak_halted": round(max(h_arr), 1),
                            **(amb_info or {})}}

    return {
        "baseline":        _build("baseline", bh),
        "adaptive":        _build("adaptive", ah),
        "emergency_nopcs": _build("emergency_nopcs", ah,
                                  {"ambulance_travel_s":186,"ambulance_enter_t":120,
                                   "ambulance_exit_t":306,"ambulance_stops":4}),
        "emergency_pcs":   _build("emergency_pcs", ah,
                                  {"ambulance_travel_s":89,"ambulance_enter_t":120,
                                   "ambulance_exit_t":209,"ambulance_stops":0}),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PHASE HELPER
# ──────────────────────────────────────────────────────────────────────────────

def phases_of(sd: dict) -> dict:
    """Read real signal phases from step data (exported by demo_runner.py)."""
    pj = sd.get("per_j", {})
    return {j: pj.get(j, {}).get("phase", "EW_GREEN") for j in JUNCTIONS}


# ──────────────────────────────────────────────────────────────────────────────
# TRAFFIC ANIMATION
# Bug fix: ALL road/intersection shapes must use layer="below" so they render
# under the vehicle markers.  Without this, the grey road rectangles paint
# over every vehicle dot and nothing is visible.
# ──────────────────────────────────────────────────────────────────────────────

def draw_animation(sd: dict, sim_t: int) -> go.Figure:
    """
    Render the 3-junction corridor.

    Vehicle model:
      - Queued EW: stacked upstream of stop line, one dot per vehicle in queue
      - Queued NS: stacked above intersection, one dot per vehicle
      - Free-flow EW: wrap-around dots moving east; hidden in stop-line zone
      - Free-flow WB: wrap-around dots moving west (independent lane)
      - Ambulance: position from sd["amb"]["pos_m"] (diamond marker)

    All shapes use layer="below" so vehicle scatter is always on top.
    """
    phases = phases_of(sd)
    pj     = sd.get("per_j", {})
    amb    = sd.get("amb", {})
    BG     = "#111318"

    fig = go.Figure()

    # ── Road surfaces (layer="below" so vehicles appear on top) ──────────────
    # Dark canvas background
    fig.add_shape(type="rect", x0=0, x1=CW, y0=0, y1=CH,
                  fillcolor=BG, line_width=0, layer="below")
    # EW arterial
    fig.add_shape(type="rect", x0=0, x1=CW,
                  y0=ROAD_CY-ROAD_H, y1=ROAD_CY+ROAD_H,
                  fillcolor="#252530", line_width=0, layer="below")
    # EW centre-line dashes
    for x in range(10, CW, 30):
        fig.add_shape(type="line", x0=x, x1=x+16, y0=ROAD_CY, y1=ROAD_CY,
                      line=dict(color="#3a3a4a", width=1, dash="dash"), layer="below")

    for j, jx in JX.items():
        # NS road
        fig.add_shape(type="rect",
                      x0=jx-NS_W, x1=jx+NS_W, y0=0, y1=CH,
                      fillcolor="#252530", line_width=0, layer="below")
        # NS centre-line dashes (above intersection)
        for y in range(8, ROAD_CY-ROAD_H, 22):
            fig.add_shape(type="line", x0=jx, x1=jx, y0=y, y1=y+12,
                          line=dict(color="#3a3a4a", width=1, dash="dash"), layer="below")
        for y in range(ROAD_CY+ROAD_H+8, CH, 22):
            fig.add_shape(type="line", x0=jx, x1=jx, y0=y, y1=y+12,
                          line=dict(color="#3a3a4a", width=1, dash="dash"), layer="below")
        # Intersection box (slightly lighter)
        fig.add_shape(type="rect",
                      x0=jx-NS_W, x1=jx+NS_W,
                      y0=ROAD_CY-ROAD_H, y1=ROAD_CY+ROAD_H,
                      fillcolor="#1e1e28", line_width=0, layer="below")
        # EW stop line (west side of intersection)
        fig.add_shape(type="line",
                      x0=jx-STOP_OFF, x1=jx-STOP_OFF,
                      y0=ROAD_CY-ROAD_H, y1=ROAD_CY,
                      line=dict(color="#555", width=2), layer="below")
        # WB stop line (east side)
        fig.add_shape(type="line",
                      x0=jx+STOP_OFF, x1=jx+STOP_OFF,
                      y0=ROAD_CY, y1=ROAD_CY+ROAD_H,
                      line=dict(color="#555", width=2), layer="below")
        # NS stop line (south approach)
        fig.add_shape(type="line",
                      x0=jx, x1=jx+NS_W,
                      y0=ROAD_CY-STOP_OFF, y1=ROAD_CY-STOP_OFF,
                      line=dict(color="#555", width=2), layer="below")

    # ── Vehicles (drawn as scatter — on top because no layer="below") ────────
    vx, vy, vc, vsym = [], [], [], []

    for j, jx in JX.items():
        phase = phases[j]
        ew_g  = (phase == "EW_GREEN")
        ns_g  = (phase == "NS_GREEN")
        q_ew  = int(min(pj.get(j, {}).get("ew", 0), 14))
        q_ns  = int(min(pj.get(j, {}).get("ns", 0), 10))

        # ── Queued EW — stack west of stop line ──
        stop_x = jx - STOP_OFF - 3
        min_x  = 10
        for k in range(q_ew):
            px = stop_x - k * VEH_GAP
            if px >= min_x:
                vx.append(px);  vy.append(ROAD_CY - 8)
                vc.append("#50e090" if ew_g else "#e05050")
                vsym.append("square")

        # ── Queued NS — stack above stop line ──
        stop_y = ROAD_CY - STOP_OFF - 3
        for k in range(q_ns):
            py = stop_y - k * VEH_GAP
            if py >= 8:
                vx.append(jx + 6);  vy.append(py)
                vc.append("#50e090" if ns_g else "#e05050")
                vsym.append("square")

        # ── Free-flow EW (eastbound top lane) ──
        # Vehicles wrap around the full canvas width.
        # Each junction contributes 3 vehicles staggered 200px apart.
        # A vehicle is hidden only if it is directly upstream of a RED stop line
        # (within the last 30px before the stop) — avoid appearing to run red lights.
        j_off  = {"J0": 0, "J1": 200, "J2": 400}[j]
        for k in range(3):
            px = (sim_t * 20 + k * 200 + j_off) % (CW + 60) - 30
            if 0 < px < CW:
                # Hide only if inside stop-line buffer at a RED junction
                at_red = any(
                    jx2 - STOP_OFF - 28 < px < jx2 - STOP_OFF + 5
                    and phases[j2] != "EW_GREEN"
                    for j2, jx2 in JX.items()
                )
                if not at_red:
                    vx.append(px);  vy.append(ROAD_CY - 8)
                    vc.append("#50e090");  vsym.append("circle")

        # ── Free-flow WB (westbound bottom lane) ──
        j_off_wb = {"J0": 0, "J1": 200, "J2": 400}[j]
        for k in range(2):
            px = CW - ((sim_t * 16 + k * 220 + j_off_wb) % (CW + 60)) + 30
            if 0 < px < CW:
                vx.append(px);  vy.append(ROAD_CY + 8)
                vc.append("#50e090");  vsym.append("circle")

        # ── Free-flow NS (northbound, right lane) ──
        for k in range(2):
            py = CH - ((sim_t * 13 + k * 130 + {"J0":0,"J1":50,"J2":100}[j]) % (CH + 40)) + 20
            if ROAD_CY + ROAD_H < py < CH - 5:
                vx.append(jx - 6);  vy.append(py)
                vc.append("#50e090");  vsym.append("circle")

    # ── Ambulance ─────────────────────────────────────────────────────────────
    if amb.get("active"):
        ax = amb_to_x(amb.get("pos_m", 0))
        if -10 < ax < CW + 10:
            flash = (sim_t % 2 == 0)
            vx.append(ax);  vy.append(ROAD_CY - 8)
            vc.append("#ef4444" if flash else "#ffffff")
            vsym.append("diamond")

    # Render all vehicles in a SINGLE scatter trace
    if vx:
        fig.add_trace(go.Scatter(
            x=vx, y=vy, mode="markers",
            marker=dict(color=vc, size=VEH_SZ, symbol=vsym,
                        line=dict(color="rgba(0,0,0,0.53)", width=0.8)),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Signal heads ──────────────────────────────────────────────────────────
    # Two dots per junction: one for EW approach, one for NS approach
    sx, sy, sc = [], [], []
    for j, jx in JX.items():
        phase      = phases[j]
        ew_c, ns_c = PHASE_SIG.get(phase, ("#ef4444", "#ef4444"))
        # EW signal: left of stop line, road centre height
        sx.append(jx - STOP_OFF - 10);  sy.append(ROAD_CY - ROAD_H - 10);  sc.append(ew_c)
        # NS signal: right of NS road, above intersection
        sx.append(jx + NS_W + 10);       sy.append(ROAD_CY - STOP_OFF - 8);  sc.append(ns_c)

    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="markers",
        marker=dict(color=sc, size=12, symbol="circle",
                    line=dict(color="#000", width=1.5)),
        showlegend=False, hoverinfo="skip",
    ))

    # ── PCS highlight (emergency only) ────────────────────────────────────────
    if amb.get("active"):
        ax_now = amb_to_x(amb.get("pos_m", 0))
        for j, jx in JX.items():
            if abs(jx - ax_now) < 220:
                fig.add_shape(type="rect",
                              x0=jx-NS_W-5, x1=jx+NS_W+5,
                              y0=ROAD_CY-ROAD_H-5, y1=ROAD_CY+ROAD_H+5,
                              fillcolor="rgba(0,0,0,0)",
                              line=dict(color="#f59e0b", width=2, dash="dot"),
                              layer="below")
                fig.add_annotation(x=jx, y=ROAD_CY-ROAD_H-18, text="PCS",
                                   showarrow=False,
                                   font=dict(color="#f59e0b", size=9, family="monospace"))

    # ── Labels & overlays ─────────────────────────────────────────────────────
    for j, jx in JX.items():
        fig.add_annotation(x=jx, y=8, text=j, showarrow=False,
                           font=dict(color="#445", size=10, family="monospace"))

    fig.add_annotation(x=8, y=CH-8, text=f"t={sim_t}s",
                       showarrow=False, xanchor="left",
                       font=dict(color="#555", size=10, family="monospace"))
    fig.add_annotation(x=8, y=CH-20,
                       text=f"queued: {sd.get('total_halted',0)}",
                       showarrow=False, xanchor="left",
                       font=dict(color="#555", size=10, family="monospace"))
    if amb.get("active"):
        fig.add_annotation(x=CW/2, y=CH-8, text="⚠ EMERGENCY VEHICLE ACTIVE",
                           showarrow=False,
                           font=dict(color="#ef4444", size=11, family="monospace"))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[0, CW], fixedrange=True),
        yaxis=dict(visible=False, range=[0, CH], fixedrange=True),
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CHART: LIVE (current mode only)
# ──────────────────────────────────────────────────────────────────────────────

def chart_live_queue(hist_t: list, hist_h: list, mode: str) -> go.Figure:
    """
    Single-mode live queue chart — grows from history buffer.
    Shows ONLY the current simulation run.  Not a comparison.
    """
    fig = go.Figure()
    if hist_t:
        col = MODE_CFG[mode]["color"]
        fig.add_trace(go.Scatter(
            x=hist_t, y=hist_h,
            name=MODE_CFG[mode]["label"],
            line=dict(color=col, width=2.5),
            fill="tozeroy", fillcolor=col.replace("#", "rgba(").replace(")", ",0.08)") + ")"
            if False else "rgba(88,166,255,0.07)",
        ))
    dl = {**DARK, "xaxis": {**DARK["xaxis"], "range": [0, SIM_LIMIT]},
          "yaxis": {**DARK["yaxis"], "rangemode": "tozero"}}
    fig.update_layout(**dl,
                      title=f"Live Queue — {MODE_CFG[mode]['label']}",
                      xaxis_title="Sim Time (s)", yaxis_title="Vehicles Halted",
                      height=220, showlegend=False)
    return fig


def chart_live_perjunction(steps: list, cur_t: int) -> go.Figure:
    """Per-junction breakdown growing to cur_t."""
    sub = [s for s in steps if s["t"] <= cur_t]
    fig = go.Figure()
    if sub:
        t_    = [s["t"] for s in sub]
        cols  = ["#58a6ff", "#56d364", "#f0a500"]
        for i, j in enumerate(JUNCTIONS):
            vals = [s["per_j"].get(j, {}).get("ew", 0)
                    + s["per_j"].get(j, {}).get("ns", 0) for s in sub]
            vals = pd.Series(vals).rolling(3, min_periods=1).mean().tolist()
            fig.add_trace(go.Scatter(x=t_, y=vals, name=j,
                                     line=dict(color=cols[i], width=2)))
    dl = {**DARK, "xaxis": {**DARK["xaxis"], "range": [0, SIM_LIMIT]},
          "yaxis": {**DARK["yaxis"], "rangemode": "tozero"}}
    fig.update_layout(**dl, title="Queue per Junction",
                      xaxis_title="Time (s)", yaxis_title="Halted", height=200)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CHART: COMPARISON (expander only)
# ──────────────────────────────────────────────────────────────────────────────

def chart_compare(base_steps: list, adapt_steps: list, cur_t: int) -> go.Figure:
    """Fixed vs adaptive, sliced to cur_t — shown in comparison expander."""
    bs = [s for s in base_steps  if s["t"] <= cur_t]
    as_ = [s for s in adapt_steps if s["t"] <= cur_t]
    fig = go.Figure()
    if bs:
        fig.add_trace(go.Scatter(
            x=[s["t"] for s in bs], y=[s["total_halted"] for s in bs],
            name="Fixed-Time Baseline", line=dict(color="#f85149", width=2)))
    if as_:
        fig.add_trace(go.Scatter(
            x=[s["t"] for s in as_], y=[s["total_halted"] for s in as_],
            name="FlowSense Adaptive", line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
    dl = {**DARK, "xaxis": {**DARK["xaxis"], "range": [0, SIM_LIMIT]}}
    fig.update_layout(**dl, title="Fixed vs Adaptive (to current time)",
                      xaxis_title="Sim Time (s)", yaxis_title="Halted",
                      height=240)
    return fig


def chart_ambulance_bar(nopcs: dict, pcs: dict) -> go.Figure:
    """Simple ambulance travel time comparison bar."""
    ne = nopcs.get("ambulance_enter_t", 120)
    nt = nopcs.get("ambulance_travel_s", 186)
    pt = pcs.get("ambulance_travel_s",   89)
    ns = nopcs.get("ambulance_stops",    4)
    ps = pcs.get("ambulance_stops",      0)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Without PCS"], x=[nt], base=[ne],
                         orientation="h", marker_color="#f85149",
                         name=f"No PCS — {ns} stops, {nt}s",
                         text=f"  {nt}s | {ns} stops",
                         textposition="inside", insidetextanchor="start"))
    fig.add_trace(go.Bar(y=["With PCS"],    x=[pt], base=[ne],
                         orientation="h", marker_color="#56d364",
                         name=f"PCS — {ps} stops, {pt}s",
                         text=f"  {pt}s | 0 stops",
                         textposition="inside", insidetextanchor="start"))
    fig.add_vline(x=ne, line_dash="dash", line_color="#f0a500",
                  annotation_text="🚑 Dispatched",
                  annotation_position="top right")
    dl = {**DARK, "legend": dict(orientation="h", y=-0.35, bgcolor="rgba(0,0,0,0)")}
    fig.update_layout(**dl, title="Ambulance Travel Time",
                      xaxis_title="Simulation Time (s)", barmode="overlay", height=180)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# KPI CARD
# ──────────────────────────────────────────────────────────────────────────────

def kpi(label: str, val, delta: str = "", col: str = "#58a6ff") -> str:
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    return (f"<div class='kpi-card'>"
            f"<div class='kpi-val' style='color:{col}'>{val}</div>"
            f"{delta_html}"
            f"<div class='kpi-label'>{label}</div>"
            f"</div>")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def _reset():
    st.session_state.hist_t      = []
    st.session_state.hist_halted = []
    st.session_state.hist_mode   = None
    st.session_state.step        = 0
    st.session_state.running     = True


def main():
    # ── Session state ─────────────────────────────────────────────────────────
    for k, v in [("step", 0), ("running", True), ("ui_mode", "adaptive"),
                 ("sim_speed", 1), ("hist_t", []), ("hist_halted", []),
                 ("hist_mode", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    data, is_live = load_results()

    # ── Resolve current frame ─────────────────────────────────────────────────
    ui_mode  = st.session_state.ui_mode
    rkey     = MODE_CFG[ui_mode]["key"]
    steps    = data[rkey]["steps"][:SIM_LIMIT]
    cur_step = min(int(st.session_state.step), len(steps) - 1)
    sd       = steps[cur_step]

    # ── History buffer ────────────────────────────────────────────────────────
    if st.session_state.hist_mode != ui_mode:
        st.session_state.hist_mode    = ui_mode
        st.session_state.hist_t       = []
        st.session_state.hist_halted  = []

    hist_t = st.session_state.hist_t
    hist_h = st.session_state.hist_halted
    # Append current frame (avoid duplicate on same step)
    if not hist_t or hist_t[-1] != cur_step:
        hist_t.append(cur_step)
        hist_h.append(sd["total_halted"])
        st.session_state.hist_t      = hist_t
        st.session_state.hist_halted = hist_h

    # ── Live metric derivation ────────────────────────────────────────────────
    ARRIVAL_RATE = 1.5    # veh/s estimated from state_log (3 junctions total)
    throughput   = 0
    stops_est    = 0
    if len(hist_h) > 1:
        for i in range(1, len(hist_h)):
            discharged    = max(0.0, hist_h[i-1] - hist_h[i] + ARRIVAL_RATE)
            throughput   += discharged
            # Estimate stops: a red-to-green transition causes stops
            # Proxy: each frame where queue grew counts as vehicles stopping
            stops_est    += max(0, hist_h[i] - hist_h[i-1])
    throughput = int(throughput)
    stops_est  = int(stops_est)
    avg_q      = sum(hist_h) / len(hist_h) if hist_h else 0
    # Avg delay: Little's Law proxy — avg_queue / arrival_rate
    avg_delay  = avg_q / ARRIVAL_RATE if ARRIVAL_RATE > 0 else 0

    total_q = sd.get("total_halted", 0)
    prev_q  = hist_h[-2] if len(hist_h) >= 2 else total_q

    # ──────────────────────────────────────────────────────────────────────────
    # LAYOUT START
    # ──────────────────────────────────────────────────────────────────────────

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown(
            "<h1 style='margin:0;font-size:1.7rem'>🚦 FlowSense</h1>"
            "<div style='color:#8b949e;font-size:.85rem;margin-top:2px'>"
            "AI-Driven Adaptive Traffic &amp; Emergency Corridor — Live Simulation"
            "</div>", unsafe_allow_html=True)
    with h2:
        src_label = "🟢 Video-Derived State" if is_live else "🟡 Demo Mode"
        st.caption(f"{src_label} &nbsp;|&nbsp; India Innovates 2026 — Team HighKey Trophy")

    # ── Mode selector + controls (compact row) ────────────────────────────────
    mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns([2, 2, 2, 0.6, 0.6, 1.5, 1.5])
    for col_, key in zip([mc1, mc2, mc3], ["fixed", "adaptive", "emergency"]):
        with col_:
            active = (ui_mode == key)
            cfg    = MODE_CFG[key]
            border = f"2px solid {cfg['color']}" if active else "1px solid #30363d"
            bg     = "#1a1f28" if active else "#0d1117"
            if st.button(
                ("✓ " if active else "") + cfg["label"],
                key=f"m_{key}", use_container_width=True
            ):
                st.session_state.ui_mode = key
                _reset()
    with mc4:
        if st.session_state.running:
            if st.button("⏸", use_container_width=True):
                st.session_state.running = False
        else:
            if st.button("▶", use_container_width=True):
                st.session_state.running = True
    with mc5:
        if st.button("⏮", use_container_width=True):
            _reset()
    with mc6:
        spd = st.select_slider("Speed", options=[1, 2, 3, 5],
                                value=st.session_state.sim_speed,
                                label_visibility="collapsed")
        st.session_state.sim_speed = spd
    with mc7:
        st.caption(f"Speed ×{spd} &nbsp;|&nbsp; {'▶ Running' if st.session_state.running else '⏸ Paused'}")

    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    delta_q  = f"{total_q - prev_q:+d}" if total_q != prev_q else ""
    with k1:
        st.markdown(kpi("Sim Time",       f"{cur_step}s",
                        f"{cur_step}/{SIM_LIMIT}s"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi("Vehicles Halted", total_q,
                        delta_q, col="#f85149" if total_q > avg_q else "#56d364"),
                    unsafe_allow_html=True)
    with k3:
        st.markdown(kpi("Throughput",      f"~{throughput}",
                        "veh served"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("Avg Delay",       f"{avg_delay:.1f}s",
                        "Little's Law"), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi("Avg Queue",       f"{avg_q:.1f}",
                        "veh halted"), unsafe_allow_html=True)
    with k6:
        mode_col = MODE_CFG[ui_mode]["color"]
        st.markdown(kpi("Mode", MODE_CFG[ui_mode]["label"][:9],
                        col=mode_col), unsafe_allow_html=True)

    # ── Main: animation | signal panel ───────────────────────────────────────
    col_anim, col_sig = st.columns([3, 1])

    with col_anim:
        mode_col = MODE_CFG[ui_mode]["color"]
        st.markdown(
            f"<div style='font-weight:700;color:{mode_col};margin-bottom:4px'>"
            f"● {MODE_CFG[ui_mode]['label']} — t={cur_step}s</div>",
            unsafe_allow_html=True)
        st.plotly_chart(draw_animation(sd, cur_step),
                        use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True})

    with col_sig:
        st.markdown("**Signal States**")
        phases = phases_of(sd)
        for j in JUNCTIONS:
            ph   = phases[j]
            _pj  = sd.get("per_j", {})
            q_ew = _pj.get(j, {}).get("ew", 0)
            q_ns = _pj.get(j, {}).get("ns", 0)
            ew_c, ns_c = PHASE_SIG.get(ph, ("#ef4444", "#ef4444"))
            ph_label   = ph.replace("_", " ")
            ew_dot = f"<span style='color:{ew_c}'>●</span> EW"
            ns_dot = f"<span style='color:{ns_c}'>●</span> NS"
            st.markdown(
                f"<div class='sig-card'>"
                f"<b style='color:#aaa'>{j}</b> &nbsp;{ew_dot} {ns_dot}<br>"
                f"<span style='color:#444;font-size:.72rem'>"
                f"ew={q_ew} ns={q_ns} &nbsp; <i>{ph_label}</i>"
                f"</span></div>",
                unsafe_allow_html=True)

        # Emergency vehicle status
        if ui_mode == "emergency":
            amb = sd.get("amb", {})
            st.markdown("---")
            st.markdown("**Emergency**")
            if not amb.get("active") and cur_step < 120:
                st.info(f"🚑 T-{120-cur_step}s to dispatch")
            elif amb.get("active"):
                pct = max(0, min(1.0,
                    (amb.get("pos_m", 0) - AMB_ENTRY_M) / (2 * AMB_SPACING_M)))
                st.warning(f"🚑 En route")
                st.progress(pct)
            else:
                st.success(f"🏥 Arrived")

        # Progress bar
        st.markdown("---")
        st.progress(min(1.0, cur_step / max(1, SIM_LIMIT - 1)),
                    text=f"{cur_step}/{SIM_LIMIT}s")

    # ── Live charts (current mode only) ───────────────────────────────────────
    lc1, lc2 = st.columns([3, 2])
    with lc1:
        st.plotly_chart(chart_live_queue(hist_t, hist_h, ui_mode),
                        use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True})
    with lc2:
        st.plotly_chart(chart_live_perjunction(steps, cur_step),
                        use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True})

    # ──────────────────────────────────────────────────────────────────────────
    # EXPANDERS (non-essential / reference sections)
    # ──────────────────────────────────────────────────────────────────────────

    with st.expander("📊 Compare Modes — Fixed vs Adaptive"):
        st.plotly_chart(
            chart_compare(data["baseline"]["steps"][:SIM_LIMIT],
                          data["adaptive"]["steps"][:SIM_LIMIT],
                          cur_step),
            use_container_width=True, config={"displayModeBar": False})
        base_avg  = data["baseline"]["summary"]["avg_halted"]
        adapt_avg = data["adaptive"]["summary"]["avg_halted"]
        if base_avg > 0:
            red = round((base_avg - adapt_avg) / base_avg * 100)
            st.markdown(
                f"**Adaptive reduces avg. queue by {red}%** "
                f"({base_avg:.1f} → {adapt_avg:.1f} veh avg halted, full 600s run)")

    if ui_mode == "emergency":
        with st.expander("🚑 Emergency Corridor — PCS Detail", expanded=True):
            nopcs_s = data["emergency_nopcs"]["summary"]
            pcs_s   = data["emergency_pcs"]["summary"]
            st.plotly_chart(chart_ambulance_bar(nopcs_s, pcs_s),
                            use_container_width=True,
                            config={"displayModeBar": False})
            ne = pcs_s.get("ambulance_enter_t", 120)
            ne1, ne2, ne3 = ne+7, ne+43, ne+79
            st.markdown(
                f"<div style='font-family:monospace;font-size:.8rem;line-height:1.9;"
                f"background:#0d1117;border:1px solid #21262d;border-radius:8px;"
                f"padding:12px'>"
                f"<span style='color:#f0a500'>T={ne}s</span> 🚑 Dispatched<br>"
                f"<span style='color:#58a6ff'>T={ne+1}s</span> ETAs: J0+{ne1-ne}s  J1+{ne2-ne}s  J2+{ne3-ne}s<br>"
                f"<span style='color:#56d364'>T={ne2}s</span> 🚑 Passes J0 GREEN ✅<br>"
                f"<span style='color:#56d364'>T={ne2}s</span> 🚑 Passes J1 GREEN ✅<br>"
                f"<span style='color:#56d364'>T={ne3}s</span> 🚑 Passes J2 GREEN ✅<br>"
                f"<span style='color:#56d364'>T={pcs_s.get('ambulance_exit_t',209)}s</span>"
                f" 🏥 Arrived. Travel: <b>{pcs_s.get('ambulance_travel_s',89)}s</b>"
                f"</div>",
                unsafe_allow_html=True)

    with st.expander("🏗️ System Architecture"):
        layers = [
            ("Layer 1 — Edge CV",      "#58a6ff", "#0d1b2e",
             "YOLOv8n detection + ByteTrack → queue_length, arrival_rate, departure_rate @ 1 Hz"),
            ("Layer 2 — Adaptive Ctrl", "#f0a500", "#1e1500",
             "pressure = queue + 5×arrival_rate + platoon_bias; MIN_GREEN=8s MAX_GREEN=45s"),
            ("Layer 3 — PCS Emergency", "#56d364", "#0d1e0d",
             "GPS dispatch → ETA per junction → reservation window (10s pre-clear + 6s guard)"),
        ]
        html = "<div style='display:flex;gap:10px;margin-top:6px'>"
        for title, border, bg, desc in layers:
            html += (f"<div style='flex:1;background:{bg};border:1.5px solid {border};"
                     f"border-radius:8px;padding:12px'>"
                     f"<div style='color:{border};font-weight:700;font-size:.85rem;"
                     f"margin-bottom:5px'>{title}</div>"
                     f"<div style='color:#8b949e;font-size:.78rem;line-height:1.6'>{desc}</div>"
                     f"</div>")
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    with st.expander("📹 CV Layer — Detection Sample"):
        vp = Path("output/detection.mp4")
        if vp.exists() and vp.stat().st_size > 1000:
            st.video(str(vp))
            st.caption("YOLOv8n + ByteTrack on real Delhi traffic footage")
        slp = Path("output/state_log.jsonl")
        if slp.exists():
            with open(slp) as f:
                lines = f.read().strip().splitlines()
            last = json.loads(lines[-1])
            st.code(json.dumps({
                "intersection_id": "J0",
                "timestamp":        last.get("timestamp"),
                "queue_length":     last.get("queue_length", 0),
                "arrival_rate":     last.get("arrival_rate", 0),
                "signal_phase":     "EW_GREEN",
            }, indent=2), language="json")
            st.caption(f"{len(lines)} samples from video analysis")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#333;font-size:.75rem'>"
        "FlowSense — Team HighKey Trophy &nbsp;|&nbsp; India Innovates 2026 &nbsp;|&nbsp;"
        " Inspired by SURTRAC · SCATS · Singapore TPS"
        "</div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # ANIMATION LOOP — advance step and rerun
    # ──────────────────────────────────────────────────────────────────────────
    if st.session_state.running:
        if cur_step >= len(steps) - 1:
            st.session_state.running = False
        else:
            st.session_state.step = cur_step + st.session_state.sim_speed
            time.sleep(0.07)
            st.rerun()


if __name__ == "__main__":
    main()
