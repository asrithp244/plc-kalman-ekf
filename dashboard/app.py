"""
Live Dash dashboard for the Virtual PLC EKF demo.

Panels:
  1. Joint angles  — raw encoder vs EKF estimate vs ground truth
  2. Joint velocities — raw gyro vs EKF estimate vs ground truth
  3. PLC register state — safety flags + NEES health indicator
  4. Safety zone visualisation — 2-DOF arm in joint space
  5. Alarm log — timestamped fault events

Run with:
    python dashboard/app.py

The dashboard reads from a shared in-process data store updated by
the main run loop (src/main.py). For the standalone demo it runs its
own internal simulation loop in a background thread.
"""

import threading
import time
import collections
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash import Dash, dcc, html, Output, Input, callback_context
import plotly.graph_objects as go

from src.ekf import RobotArmEKF, check_safety, JOINT_LIMIT_RAD, VELOCITY_LIMIT
from src.simulator import RobotArmSimulator, sinusoidal_torque, velocity_ramp_torque

# ── Shared ring buffer ─────────────────────────────────────────────────────────
HISTORY = 500   # data points to keep (~5s at 100Hz, or 50s at 10Hz)

_lock = threading.Lock()
_data: dict[str, collections.deque] = {k: collections.deque(maxlen=HISTORY) for k in [
    "t",
    "q1_true", "q1_noisy", "q1_ekf",
    "q2_true", "q2_noisy", "q2_ekf",
    "qd1_true", "qd1_noisy", "qd1_ekf",
    "qd2_true", "qd2_noisy", "qd2_ekf",
    "nees",
    "j1_limit", "j2_limit", "j1_vel", "j2_vel", "any_fault",
]}
_alarms: collections.deque = collections.deque(maxlen=50)
_sim_running = True


# ── Background simulation thread ───────────────────────────────────────────────

def _simulation_loop():
    """
    Runs EKF + simulator in background, populates _data at ~100Hz.
    Switches torque profile after 15s to trigger velocity interlock.
    """
    ekf = RobotArmEKF()
    sim = RobotArmSimulator(dt=0.01, seed=42)
    state = ekf.init()

    def torque_fn(t, q, qd):
        if t < 15.0:
            return sinusoidal_torque(t, q, qd)
        return velocity_ramp_torque(t, q, qd)

    x_true_prev = np.zeros(4)

    for true, sensors in sim.run(duration=120.0, torque_fn=torque_fn):
        if not _sim_running:
            break

        state = ekf.predict(state, true.tau, dt=0.01)
        if not sensors.enc_fault:
            state = ekf.update_encoder(state, sensors.enc)
        if not sensors.gyro_fault:
            state = ekf.update_imu(state, sensors.gyro)

        x_true = np.array([true.q[0], true.qd[0], true.q[1], true.qd[1]])
        nees   = ekf.nees(state, x_true)
        flags  = check_safety(state)

        with _lock:
            _data["t"].append(true.t)
            _data["q1_true"].append(np.degrees(true.q[0]))
            _data["q1_noisy"].append(np.degrees(sensors.enc[0]) if not sensors.enc_fault else np.nan)
            _data["q1_ekf"].append(np.degrees(state.x[0]))
            _data["q2_true"].append(np.degrees(true.q[1]))
            _data["q2_noisy"].append(np.degrees(sensors.enc[1]) if not sensors.enc_fault else np.nan)
            _data["q2_ekf"].append(np.degrees(state.x[1]))
            _data["qd1_true"].append(true.qd[0])
            _data["qd1_noisy"].append(sensors.gyro[0] if not sensors.gyro_fault else np.nan)
            _data["qd1_ekf"].append(state.x[1])
            _data["qd2_true"].append(true.qd[1])
            _data["qd2_noisy"].append(sensors.gyro[1] if not sensors.gyro_fault else np.nan)
            _data["qd2_ekf"].append(state.x[3])
            _data["nees"].append(min(nees, 100.0))
            for k, v in flags.items():
                _data[k].append(int(v))

            # Alarm log
            if flags["any_fault"]:
                active = [k for k, v in flags.items() if v and k != "any_fault"]
                _alarms.append(f"[{true.t:6.2f}s] FAULT: {', '.join(active)}")

        time.sleep(0.01)


# ── Dash app ───────────────────────────────────────────────────────────────────

app = Dash(__name__, title="Virtual PLC — EKF Dashboard")

DARK_BG  = "#1a1a2e"
CARD_BG  = "#16213e"
ACCENT   = "#0f3460"
GREEN    = "#00b894"
RED      = "#d63031"
YELLOW   = "#fdcb6e"
BLUE     = "#74b9ff"
WHITE    = "#dfe6e9"

def _card(children, style=None):
    base = {
        "background": CARD_BG,
        "borderRadius": "8px",
        "padding": "12px",
        "marginBottom": "12px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


app.layout = html.Div(style={"background": DARK_BG, "minHeight": "100vh",
                              "fontFamily": "monospace", "color": WHITE,
                              "padding": "16px"}, children=[

    html.H2("Virtual PLC — EKF Joint State Monitor",
            style={"color": GREEN, "marginBottom": "4px"}),
    html.P("2-DOF Robot Arm | Extended Kalman Filter | IEC 61131-3 Safety Interlock",
           style={"color": "#aaa", "marginTop": 0, "marginBottom": "16px"}),

    # Status bar
    _card([
        html.Div(id="status-bar", style={"display": "flex", "gap": "24px"})
    ], style={"padding": "8px 12px"}),

    # Row 1: angle plots
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
        _card([html.H4("Joint 1 Angle [°]", style={"margin": "0 0 8px 0", "color": GREEN}),
               dcc.Graph(id="plot-q1", style={"height": "220px"})]),
        _card([html.H4("Joint 2 Angle [°]", style={"margin": "0 0 8px 0", "color": GREEN}),
               dcc.Graph(id="plot-q2", style={"height": "220px"})]),
    ]),

    # Row 2: velocity plots
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
        _card([html.H4("Joint 1 Velocity [rad/s]", style={"margin": "0 0 8px 0", "color": BLUE}),
               dcc.Graph(id="plot-qd1", style={"height": "220px"})]),
        _card([html.H4("Joint 2 Velocity [rad/s]", style={"margin": "0 0 8px 0", "color": BLUE}),
               dcc.Graph(id="plot-qd2", style={"height": "220px"})]),
    ]),

    # Row 3: NEES + safety zone + alarm log
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"}, children=[
        _card([html.H4("Filter Health (NEES)", style={"margin": "0 0 8px 0", "color": YELLOW}),
               dcc.Graph(id="plot-nees", style={"height": "220px"})]),
        _card([html.H4("Joint Space Safety Zone", style={"margin": "0 0 8px 0", "color": YELLOW}),
               dcc.Graph(id="plot-safety-zone", style={"height": "220px"})]),
        _card([html.H4("Alarm Log", style={"margin": "0 0 8px 0", "color": RED}),
               html.Div(id="alarm-log", style={"height": "200px", "overflowY": "auto",
                                                "fontSize": "11px", "color": RED})]),
    ]),

    dcc.Interval(id="interval", interval=100, n_intervals=0),  # 10 Hz refresh
])


# ── Callbacks ──────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=WHITE, size=10),
    margin=dict(l=40, r=10, t=10, b=30),
    legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    xaxis=dict(showgrid=True, gridcolor="#2d3436", title="t [s]"),
    yaxis=dict(showgrid=True, gridcolor="#2d3436"),
)


def _snapshot():
    with _lock:
        return {k: list(v) for k, v in _data.items()}, list(_alarms)


@app.callback(
    Output("plot-q1",          "figure"),
    Output("plot-q2",          "figure"),
    Output("plot-qd1",         "figure"),
    Output("plot-qd2",         "figure"),
    Output("plot-nees",        "figure"),
    Output("plot-safety-zone", "figure"),
    Output("alarm-log",        "children"),
    Output("status-bar",       "children"),
    Input("interval",          "n_intervals"),
)
def update_all(_):
    d, alarms = _snapshot()
    t = d["t"]

    def line_fig(series_dict: dict, limit_line: float = None, limit_color=RED):
        fig = go.Figure()
        colors = {"Ground Truth": WHITE, "Raw Sensor": YELLOW, "EKF Estimate": GREEN}
        for name, y in series_dict.items():
            fig.add_trace(go.Scatter(
                x=t, y=y, name=name, mode="lines",
                line=dict(color=colors.get(name, WHITE),
                          width=1 if name != "EKF Estimate" else 2,
                          dash="dot" if name == "Raw Sensor" else "solid")
            ))
        if limit_line is not None:
            for sign in [1, -1]:
                fig.add_hline(y=sign * limit_line,
                              line_dash="dash", line_color=limit_color, line_width=1)
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    # Angle plots
    j1_lim_deg = np.degrees(JOINT_LIMIT_RAD[0])
    j2_lim_deg = np.degrees(JOINT_LIMIT_RAD[1])

    fig_q1 = line_fig({"Ground Truth": d["q1_true"],
                        "Raw Sensor":   d["q1_noisy"],
                        "EKF Estimate": d["q1_ekf"]}, j1_lim_deg)
    fig_q2 = line_fig({"Ground Truth": d["q2_true"],
                        "Raw Sensor":   d["q2_noisy"],
                        "EKF Estimate": d["q2_ekf"]}, j2_lim_deg)

    # Velocity plots
    fig_qd1 = line_fig({"Ground Truth": d["qd1_true"],
                         "Raw Sensor":   d["qd1_noisy"],
                         "EKF Estimate": d["qd1_ekf"]}, VELOCITY_LIMIT[0])
    fig_qd2 = line_fig({"Ground Truth": d["qd2_true"],
                         "Raw Sensor":   d["qd2_noisy"],
                         "EKF Estimate": d["qd2_ekf"]}, VELOCITY_LIMIT[1])

    # NEES plot
    fig_nees = go.Figure()
    fig_nees.add_trace(go.Scatter(x=t, y=d["nees"], name="NEES",
                                   mode="lines", line=dict(color=YELLOW, width=1.5)))
    fig_nees.add_hline(y=4.0,  line_dash="dash", line_color=GREEN,   line_width=1,
                        annotation_text="Expected (χ²=4)")
    fig_nees.add_hline(y=18.5, line_dash="dash", line_color=RED,     line_width=1,
                        annotation_text="99.9% limit")
    fig_nees.update_layout(**PLOT_LAYOUT)

    # Safety zone: joint space scatter
    fig_zone = go.Figure()
    q1_ekf_rad = [np.radians(v) for v in d["q1_ekf"]]
    q2_ekf_rad = [np.radians(v) for v in d["q2_ekf"]]
    colors_scatter = [RED if f else GREEN for f in d["any_fault"]]
    fig_zone.add_trace(go.Scatter(
        x=q1_ekf_rad, y=q2_ekf_rad, mode="markers",
        marker=dict(color=colors_scatter, size=3),
        name="EKF trajectory"
    ))
    # Draw safe zone rectangle
    lim = JOINT_LIMIT_RAD
    fig_zone.add_shape(type="rect",
        x0=-lim[0], y0=-lim[1], x1=lim[0], y1=lim[1],
        line=dict(color=GREEN, width=1, dash="dash"),
        fillcolor="rgba(0,184,148,0.05)")
    fig_zone.update_layout(**PLOT_LAYOUT,
                            xaxis=dict(showgrid=True, gridcolor="#2d3436", title="q1 [rad]"),
                            yaxis=dict(showgrid=True, gridcolor="#2d3436", title="q2 [rad]"))

    # Alarm log
    alarm_lines = [html.Div(a) for a in reversed(alarms)] if alarms else [
        html.Div("No alarms", style={"color": GREEN})]

    # Status bar
    if d["any_fault"]:
        last_fault = d["any_fault"][-1]
        fault_color = RED if last_fault else GREEN
        fault_text  = "⚠ FAULT ACTIVE" if last_fault else "✓ NOMINAL"
    else:
        fault_color, fault_text = GREEN, "✓ NOMINAL"

    nees_last = d["nees"][-1] if d["nees"] else 0.0
    status_items = [
        html.Span(fault_text, style={"color": fault_color, "fontWeight": "bold"}),
        html.Span(f"NEES: {nees_last:.1f}", style={"color": YELLOW}),
        html.Span(f"t = {t[-1]:.2f}s" if t else "t = 0.00s", style={"color": WHITE}),
        html.Span(f"Points: {len(t)}", style={"color": "#aaa"}),
    ]

    return (fig_q1, fig_q2, fig_qd1, fig_qd2, fig_nees,
            fig_zone, alarm_lines, status_items)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim_thread = threading.Thread(target=_simulation_loop, daemon=True, name="sim")
    sim_thread.start()
    print("Dashboard: http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
