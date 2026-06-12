"""ENG3032 - Welding Robot Lab (Streamlit version).

Kinematics -> Dynamics -> Power -> Energy for a single-joint welding robot arm.
Maths identical to the course MATLAB model: trapezoidal integration, 501 samples.

    th1(t) = 0.02 t^2 + 0.20 t
    th2(t) = 0.01 t^3 + 0.05 t          0 <= t <= 5 s
    Q = m l^2 th'' + C th' + (m g l / 2) cos(th)
    P = Q th'      Ee = integral |P| dt

Run with:  streamlit run welding_robot_lab.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="ENG3032 · Welding Robot Lab", page_icon="🦾", layout="wide")

# cap and centre the content on ultra-wide displays — no dead space on the right
st.markdown("<style>.block-container{max-width:1750px;margin:0 auto;}</style>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- model
G = 9.81
PAR = {}                       # per-scenario parameters {m, l, C, a, b} — filled from the sidebar
CYCLES, PRICE, CO2_G, MJ3 = 9e5, 0.45, 124.0, 3e6
N = 501
TV = np.linspace(0.0, 5.0, N)
TVS = TV[::4]                   # down-sampled paths keep the figure light
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------- theme palette
_theme = getattr(st.context, "theme", None)
DARK = getattr(_theme, "type", "light") == "dark"

if DARK:
    INK, DIM = "#e9eff6", "#9aabbd"
    PAPER, BLUEPRINT = "#101828", "#0e1a2f"
    GRID, GRIDMAJ = "#1c2d4b", "#293d63"
    CH_GRID, CH_GRIDMIN = "#21304f", "#182742"
    CARD_BG, CARD_BORDER = "#1c2638", "#2b3a55"
    ANN_BG, DATUM = "rgba(13,20,34,0.92)", "#46587a"
    PATH_FAINT, NOWLINE = "#3a4c6b", "#46587a"
    S1, S2 = "#ff6e57", "#6fa9ff"
    COL_IN, COL_DA, COL_GR = "#ff6e57", "#6fa9ff", "#46c47c"
else:
    INK, DIM = "#16222e", "#5d6b78"
    PAPER, BLUEPRINT = "white", "#eef4fb"
    GRID, GRIDMAJ = "#dae6f5", "#b3c9e6"
    CH_GRID, CH_GRIDMIN = "#dde4ea", "#eef2f6"
    CARD_BG, CARD_BORDER = "#f3f6f9", "#d5dde4"
    ANN_BG, DATUM = "rgba(255,255,255,0.92)", "#9aa8b4"
    PATH_FAINT, NOWLINE = "#c4cfd9", "#b9c4ce"
    S1, S2 = "#c8331f", "#1f5fc8"
    COL_IN, COL_DA, COL_GR = "#c8331f", "#1f5fc8", "#1f7a3d"
DARKLINE = "#1f262d"


def kin(which, t):
    """Joint angle, angular velocity and acceleration at time t (per-scenario profile)."""
    p = PAR[which]
    a, b = p["a"], p["b"]
    t = np.asarray(t, dtype=float)
    if which == 1:                                   # theta1 = a t^2 + b t
        return a * t**2 + b * t, 2 * a * t + b, np.full_like(t, 2 * a)
    return a * t**3 + b * t, 3 * a * t**2 + b, 6 * a * t


def dyn(which, t):
    """Full state from that scenario's own m, l, C: kinematics, torques, power."""
    p = PAR[which]
    m, l, C = p["m"], p["l"], p["C"]
    th, d, dd = kin(which, t)
    inertia = m * l * l * dd
    damp = C * d
    grav = m * G * l / 2 * np.cos(th)
    Q = inertia + damp + grav
    return dict(th=th, d=d, dd=dd, x=l * np.cos(th), y=l * np.sin(th),
                inertia=inertia, damp=damp, grav=grav, Q=Q, P=Q * d)


def at(which, t):
    """State at a single instant, as plain floats."""
    return {k: float(v) for k, v in dyn(which, t).items()}


@st.cache_data
def _series_cached(which, m, l, C, a, b):
    return dyn(which, TV)


def series(which):
    """Full 501-sample time series (cached per scenario parameter set)."""
    p = PAR[which]
    return _series_cached(which, p["m"], p["l"], p["C"], p["a"], p["b"])


def energy(P):
    """Electrical energy per cycle, Ee = trapz(|P|, t)  [J]."""
    return float(TRAPZ(np.abs(P), TV))


def fmt(v, d=0):
    return f"{v:,.{d}f}"


# ---------------------------------------------------------------- sidebar
def param(label, lo, hi, default, step, key):
    """Slider + numeric entry box, kept in sync — drag or type either one."""
    skey, nkey = key + "_sld", key + "_num"
    st.session_state.setdefault(skey, default)
    st.session_state.setdefault(nkey, default)

    def _from_slider():
        st.session_state[nkey] = st.session_state[skey]

    def _from_number():
        st.session_state[skey] = st.session_state[nkey]

    c1, c2 = st.columns([2.5, 1.2], vertical_alignment="bottom")
    with c1:
        st.slider(label, lo, hi, step=step, key=skey, on_change=_from_slider)
    with c2:
        st.number_input(label, lo, hi, step=step, key=nkey, on_change=_from_number,
                        label_visibility="collapsed")
    return float(st.session_state[skey])


with st.sidebar:
    st.title("🦾 Parameters")
    st.caption("Drag a slider **or type a value** in the box next to it — the formulas, "
               "the schematic and the charts all follow.")
    view = st.selectbox("Robot view", ["Compare both", "Scenario 1", "Scenario 2"], index=2)
    t = param("Time t [s]", 0.0, 5.0, 0.0, 0.01, "t")
    st.caption("▶ Play animates the schematic **and** the charts together. "
               "This slider sets the instant used by the numbers on the right.")
    st.divider()

    def scenario_box(w, title, eq, a_def, b_def, a_unit):
        """One scenario's complete, independent parameter set."""
        with st.expander(title, expanded=True):
            st.latex(eq)
            return dict(
                m=param(f"Mass m [kg] · S{w}", 50.0, 700.0, 250.0, 1.0, f"m{w}"),
                l=param(f"Arm length l [m] · S{w}", 0.5, 2.0, 1.2, 0.01, f"l{w}"),
                C=param(f"Damping C [Nm·s/rad] · S{w}", 0.0, 300.0, 100.0, 1.0, f"C{w}"),
                a=st.number_input(f"a [{a_unit}] · S{w}", 0.0, 1.0, a_def, 0.005,
                                  key=f"a{w}", format="%.3f"),
                b=st.number_input(f"b [rad/s] · S{w}", 0.0, 2.0, b_def, 0.01,
                                  key=f"b{w}", format="%.2f"),
            )

    PAR[1] = scenario_box(1, "🔴 Scenario 1 — parameters",
                          r"\theta_1(t) = a\,t^2 + b\,t", 0.02, 0.20, "rad/s²")
    PAR[2] = scenario_box(2, "🔵 Scenario 2 — parameters",
                          r"\theta_2(t) = a\,t^3 + b\,t", 0.01, 0.05, "rad/s³")
    st.caption("Course defaults — S1: a=0.02, b=0.20 · S2: a=0.01, b=0.05 · "
               "m=250 kg, l=1.2 m, C=100")
    charts = st.selectbox("Bottom charts",
                          ["Position: X & Y", "Trajectory: Y vs X", "Angle & Power", "Torque & Power"])
    st.divider()
    show_annual = st.toggle("📊 Annual fleet analysis", value=False,
                            help="Optional: energy, cost and CO₂ for a whole robot fleet")
    if show_annual:
        R = param("Robot fleet [units]", 1.0, 2000.0, 150.0, 1.0, "R")
        cyc = st.number_input("Cycles / robot / year", 1_000.0, 10_000_000.0,
                              CYCLES, 10_000.0, key="cyc")
        price = st.number_input("Electricity price [£ per 3 MJ]", 0.01, 10.0,
                                PRICE, 0.01, key="price")
        co2g = st.number_input("CO₂ intensity [g per 3 MJ]", 1.0, 2000.0,
                               CO2_G, 1.0, key="co2g")
        st.caption(f"{cyc:,.0f} cycles/robot/year · £{price:.2f} and "
                   f"{co2g:.0f} g CO₂ per 3 MJ")
    else:
        R, cyc, price, co2g = 150.0, CYCLES, PRICE, CO2_G

# ---------------------------------------------------------------- compute
LR = max(PAR[1]["l"], PAR[2]["l"])        # drawing scale: longest arm
f1, f2 = series(1), series(2)
s1, s2 = at(1, t), at(2, t)
E1, E2 = energy(f1["P"]), energy(f2["P"])
c1, c2 = (E * cyc / MJ3 * price * R for E in (E1, E2))
g1, g2 = (E * cyc / MJ3 * co2g * R for E in (E1, E2))

both = view == "Compare both"
SCEN = [1, 2] if both else ([1] if view == "Scenario 1" else [2])
single = not both
fs = s1 if view == "Scenario 1" else s2
Es = E1 if view == "Scenario 1" else E2
cs = c1 if view == "Scenario 1" else c2
gs = g1 if view == "Scenario 1" else g2

# ---------------------------------------------------------------- header + calculation chain
st.title("Welding Robot Lab")
st.caption("ENG3032 · Sustainable Systems & Industry 4.0 — single-joint arm · "
           "real Python (NumPy) calculations, identical to the course MATLAB model")


def chain_node(label, formula, v1, v2=None):
    second = f"<div style='color:{S2};font-family:monospace;font-size:13px;font-weight:600'>{v2}</div>" if v2 else ""
    vcol = S1 if v2 else INK
    return (f"<div style='border:1px solid {CARD_BORDER};border-bottom:4px solid #e8920a;"
            f"border-radius:8px;padding:9px 12px;background:{CARD_BG}'>"
            f"<div style='font-size:10.5px;letter-spacing:1.3px;color:{DIM};"
            f"text-transform:uppercase'>{label}</div>"
            f"<div style='font-family:monospace;font-size:17px;font-weight:600'>{formula}</div>"
            f"<div style='color:{vcol};font-family:monospace;font-size:13px;font-weight:600'>{v1}</div>{second}</div>")


if both:
    nodes = [
        ("Input", "θ(t)", f"S1 {s1['th']:.2f} rad", f"S2 {s2['th']:.2f} rad"),
        ("Derivative", "θ̇ , θ̈", f"S1 {s1['d']:.2f} · {s1['dd']:.3f}", f"S2 {s2['d']:.2f} · {s2['dd']:.3f}"),
        ("Dynamics", "Q", f"S1 {fmt(s1['Q'])} Nm", f"S2 {fmt(s2['Q'])} Nm"),
        ("× θ̇", "P", f"S1 {fmt(s1['P'])} W", f"S2 {fmt(s2['P'])} W"),
        ("∫|P|dt", "Eₑ /cycle", f"S1 {E1 / 1000:.2f} kJ", f"S2 {E2 / 1000:.2f} kJ"),
        ("× Fleet/yr", "£ · CO₂", f"S1 £{fmt(c1)} · {g1 / 1e6:.1f} t", f"S2 £{fmt(c2)} · {g2 / 1e6:.1f} t"),
    ]
else:
    nodes = [
        ("Input", "θ(t)", f"{fs['th']:.2f} rad", None),
        ("Derivative", "θ̇ , θ̈", f"{fs['d']:.2f} rad/s · {fs['dd']:.3f} rad/s²", None),
        ("Dynamics", "Q", f"{fmt(fs['Q'])} Nm", None),
        ("× θ̇", "P", f"{fmt(fs['P'])} W", None),
        ("∫|P|dt", "Eₑ /cycle", f"{Es / 1000:.2f} kJ", None),
        ("× Fleet/yr", "£ · CO₂", f"£{fmt(cs)} · {gs / 1e6:.1f} t", None),
    ]

if not show_annual:
    nodes = nodes[:-1]                       # fleet economics is opt-in
for col, n in zip(st.columns(len(nodes)), nodes):
    col.markdown(chain_node(*n), unsafe_allow_html=True)
st.caption("These cards follow the **t** slider — during ▶ Play the same values run live "
           "inside the schematic (top-left HUD).")

st.write("")

# ---------------------------------------------------------------- geometry helpers
def rot(px, py, th):
    px, py = np.asarray(px, float), np.asarray(py, float)
    c, s = np.cos(th), np.sin(th)
    return c * px - s * py, s * px + c * py


def arc(cx, cy, r, a0, a1, n=16):
    a = np.linspace(a0, a1, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def poly(xs, ys, fill, line=DARKLINE, w=1.6, ax="x", ay="y"):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return go.Scatter(x=np.append(xs, xs[0]), y=np.append(ys, ys[0]), mode="lines",
                      fill="toself", fillcolor=fill, line=dict(color=line, width=w),
                      xaxis=ax, yaxis=ay, hoverinfo="skip", showlegend=False)


def sc(ax="x", ay="y", **kw):
    kw.setdefault("hoverinfo", "skip")
    kw.setdefault("showlegend", False)
    return go.Scatter(xaxis=ax, yaxis=ay, **kw)


# ---------------------------------------------------------------- schematic pieces
def schematic_static():
    """Blueprint-style base: yellow pedestal block + grey hub semicircle + faint paths."""
    T = [
        poly(np.array([-0.14, 0.14, 0.14, -0.14]) * LR,
             np.array([0.0, 0.0, -0.34, -0.34]) * LR, "#b9a02d", "#5d5414"),
        poly(*arc(0, 0, 0.085 * LR, 0, np.pi, 24), "#c9ced4", "#5d5414"),
    ]
    for w in SCEN:
        f = series(w)
        T.append(sc(x=f["x"][::4], y=f["y"][::4], mode="lines", opacity=0.25,
                    line=dict(color=S1 if w == 1 else S2, width=1, dash="dot")))
    return T


def arm_anim(ft, w):
    """One arm at time ft — blueprint schematic: thick link, tip dot, live coordinates."""
    f = series(w)
    s = at(w, ft)
    col = S1 if w == 1 else S2
    tipx, tipy = s["x"], s["y"]
    mask = TVS <= ft + 1e-9
    T = [
        # traced tip path
        sc(x=f["x"][::4][mask], y=f["y"][::4][mask], mode="lines",
           line=dict(color=col, width=2)),
        # the arm: single thick link from the joint to the tip
        sc(x=[0, tipx], y=[0, tipy], mode="lines", line=dict(color=col, width=9)),
        sc(x=[0], y=[0], mode="markers",
           marker=dict(size=11, color=col)),                      # round cap at the hub
        sc(x=[0], y=[0], mode="markers",
           marker=dict(size=6, color="#c9ced4",
                       line=dict(color="#5d5414", width=1))),     # joint pin
        # tip dot — live coordinates are shown as a callout annotation (see build_figure)
        sc(x=[tipx], y=[tipy], mode="markers",
           marker=dict(size=12, color=col, line=dict(color="#16222e", width=1.5))),
    ]
    # weld sparks at the tip
    rng = np.random.default_rng(int(round(ft * 100)) + w + 1)
    nsp = 0 if ft == 0 else 8
    T.append(sc(x=tipx + rng.normal(0, 0.035 * LR, nsp),
                y=tipy + rng.normal(-0.01 * LR, 0.035 * LR, nsp),
                mode="markers", opacity=0.9,
                marker=dict(symbol="star", color="#ff9d1f", size=rng.uniform(4, 9, nsp))))
    return T


def gauge_anim(ft):
    """Single-scenario annotations: θ arc, axis projections, gravity arrow at l/2."""
    s = at(SCEN[0], ft)
    lg = PAR[SCEN[0]]["l"]
    th, x, y = s["th"], s["x"], s["y"]
    aa = np.linspace(0.0, max(th, 0.02), 30)
    gx, gy = lg / 2 * np.cos(th), lg / 2 * np.sin(th)
    return [
        sc(x=0.30 * lg * np.cos(aa), y=0.30 * lg * np.sin(aa), mode="lines",
           line=dict(color=COL_IN, width=2)),
        sc(x=[0.46 * lg * np.cos(th / 2)], y=[0.46 * lg * np.sin(th / 2)], mode="text",
           text=[f"θ = {th:.2f} rad"],
           textfont=dict(color=COL_IN, size=12.5, family="IBM Plex Mono, monospace")),
        # dashed projections from the tip onto the X and Y axes
        sc(x=[x, x, np.nan, x, 0], y=[y, 0, np.nan, y, y], mode="lines",
           line=dict(color=DIM, width=1, dash="dot")),
        sc(x=[gx, gx], y=[gy, gy - 0.26 * lg], mode="lines",
           line=dict(color=COL_GR, width=2.5)),
        sc(x=[gx], y=[gy - 0.26 * lg], mode="markers+text", text=["mg"],
           textposition="bottom center", textfont=dict(color=COL_GR, size=12),
           marker=dict(symbol="triangle-down", size=11, color=COL_GR)),
    ]


# ---------------------------------------------------------------- chart pieces
TIME_MODES = {
    "Position: X & Y": (("x", "X [m]", "X-position vs time"), ("y", "Y [m]", "Y-position vs time")),
    "Angle & Power": (("th", "θ [rad]", "Joint angle vs time"), ("P", "P [W]", "Electrical power vs time")),
    "Torque & Power": (("Q", "Q [Nm]", "Required torque vs time"), ("P", "P [W]", "Electrical power vs time")),
}
TRAJ_MODE = charts == "Trajectory: Y vs X"


def yrange(key):
    lo = min(f1[key].min(), f2[key].min())
    hi = max(f1[key].max(), f2[key].max())
    mrg = (hi - lo) * 0.08 or 1.0
    return lo - mrg, hi + mrg


def charts_static():
    T = []
    if TRAJ_MODE:
        for w, ax, ay in ((1, "x2", "y2"), (2, "x3", "y3")):
            f = series(w)
            T.append(sc(ax, ay, x=f["x"], y=f["y"], mode="lines",
                        line=dict(color=PATH_FAINT, width=1.5, dash="dot")))
            T.append(sc(ax, ay, x=[f["x"][0], f["x"][-1]], y=[f["y"][0], f["y"][-1]],
                        mode="markers+text", text=["initial", "final"],
                        textposition="middle right", textfont=dict(size=11, color=DIM),
                        marker=dict(size=8, color=[COL_GR, COL_IN],
                                    line=dict(color="#16222e", width=1))))
    else:
        (kL, _, _), (kR, _, _) = TIME_MODES[charts]
        for key, ax, ay in ((kL, "x2", "y2"), (kR, "x3", "y3")):
            T.append(sc(ax, ay, x=TV, y=f1[key], mode="lines", name="Scenario 1",
                        showlegend=(ax == "x2"), hoverinfo="x+y",
                        line=dict(color=S1, width=2)))
            T.append(sc(ax, ay, x=TV, y=f2[key], mode="lines", name="Scenario 2",
                        showlegend=(ax == "x2"), hoverinfo="x+y",
                        line=dict(color=S2, width=2)))
    return T


def charts_anim(ft):
    T = []
    if TRAJ_MODE:
        mask = TV <= ft + 1e-9
        for w, ax, ay in ((1, "x2", "y2"), (2, "x3", "y3")):
            f = series(w)
            s = at(w, ft)
            col = S1 if w == 1 else S2
            T.append(sc(ax, ay, x=f["x"][mask][::4], y=f["y"][mask][::4], mode="lines",
                        line=dict(color=col, width=2.5)))
            T.append(sc(ax, ay, x=[s["x"]], y=[s["y"]], mode="markers",
                        marker=dict(size=10, color=col, line=dict(color="#16222e", width=1))))
    else:
        (kL, _, _), (kR, _, _) = TIME_MODES[charts]
        sA, sB = at(1, ft), at(2, ft)
        for key, ax, ay in ((kL, "x2", "y2"), (kR, "x3", "y3")):
            ylo, yhi = yrange(key)
            T.append(sc(ax, ay, x=[ft, ft], y=[ylo, yhi], mode="lines",
                        line=dict(color=NOWLINE, width=1.5, dash="dash")))
            T.append(sc(ax, ay, x=[ft, ft], y=[sA[key], sB[key]], mode="markers",
                        marker=dict(color=[S1, S2], size=10,
                                    line=dict(color="#16222e", width=1))))
    return T


# ---------------------------------------------------------------- assemble the animated figure
def anim_data(ft):
    T = []
    for w in SCEN:
        T += arm_anim(ft, w)
    if single:
        T += gauge_anim(ft)
    return T + charts_anim(ft)


def build_figure():
    if TRAJ_MODE:
        titles = ("", "Scenario 1 — trajectory", "Scenario 2 — trajectory")
    else:
        titles = ("", TIME_MODES[charts][0][2], TIME_MODES[charts][1][2])
    # fixed pixel geometry: big 1:1 schematic, always-square bottom charts, and a
    # timeline whose width is computed to line up exactly with the schematic
    W, ROW1, ROW2, VSP = 1120, 660, 500, 70
    MT, MB, ML, MR = 100, 84, 10, 10
    inner_h = ROW1 + ROW2 + VSP
    H = inner_h + MT + MB
    aspect = (1.60 + 0.85) / (1.45 + 0.55)            # schematic x-span / y-span
    xdom = min(1.0, ROW1 * aspect / (W - ML - MR))    # schematic domain width (paper frac)
    x0 = (1 - xdom) / 2
    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                        row_heights=[ROW1 / (ROW1 + ROW2), ROW2 / (ROW1 + ROW2)],
                        vertical_spacing=VSP / inner_h,
                        horizontal_spacing=0.09, subplot_titles=titles)

    static = schematic_static() + charts_static()
    fig.add_traces(static)
    first = anim_data(t)
    fig.add_traces(first)
    idx = list(range(len(static), len(static) + len(first)))

    # subplot titles are annotations — style them, then keep them as the base layer
    for a in fig.layout.annotations:
        a.font = dict(size=13, color=INK)
    base_ann = [a.to_plotly_json() for a in fig.layout.annotations]

    def callouts(ft):
        """Per-frame annotations: tip callout, live HUD chain and the l dimension label."""
        ann = []
        for k, w in enumerate(SCEN):
            s = at(w, ft)
            col = S1 if w == 1 else S2
            ann.append(dict(x=s["x"], y=s["y"], xref="x", yref="y", showarrow=True,
                            text=f"X={s['x']:.2f}  Y={s['y']:.2f} m",
                            font=dict(color=col, size=12, family="IBM Plex Mono, monospace"),
                            bgcolor=ANN_BG, bordercolor=col,
                            borderwidth=1.2, borderpad=3, arrowcolor=col, arrowwidth=1.2,
                            ax=72, ay=46 if single else (42 if w == 1 else 90)))
            # live HUD: pinned to the empty sky band at the very top — the tip
            # callouts open downwards, so they can never collide with it
            ann.append(dict(xref="x domain", yref="y domain", x=0.01, y=0.995 - 0.068 * k,
                            xanchor="left", yanchor="top", showarrow=False, align="left",
                            text=(f"S{w} · θ={s['th']:.2f} rad · θ̇={s['d']:.2f} rad/s · "
                                  f"θ̈={s['dd']:.3f} rad/s² · Q={s['Q']:,.0f} Nm · "
                                  f"P={s['P']:,.0f} W"),
                            font=dict(color=col, size=12, family="IBM Plex Mono, monospace"),
                            bgcolor=ANN_BG, bordercolor=col,
                            borderwidth=1.2, borderpad=4))
        if single:
            th = at(SCEN[0], ft)["th"]
            lg = PAR[SCEN[0]]["l"]
            ann.append(dict(x=0.60 * lg * np.cos(th), y=0.60 * lg * np.sin(th),
                            xref="x", yref="y", showarrow=False,
                            text=f"l = {lg:.2f} m", textangle=float(-np.degrees(th)),
                            xshift=float(-16 * np.sin(th)), yshift=float(16 * np.cos(th)),
                            font=dict(size=11, color=DIM,
                                      family="IBM Plex Mono, monospace")))
        return ann

    fts = np.linspace(0.0, 5.0, 51)
    fig.frames = [go.Frame(data=anim_data(ft), traces=idx, name=f"{ft:.1f}",
                           layout=dict(annotations=base_ann + callouts(ft)))
                  for ft in fts]

    fig.update_layout(
        annotations=base_ann + callouts(t),
        updatemenus=[dict(type="buttons", direction="left", x=x0, xanchor="left",
                          y=1.075, showactive=False, pad=dict(r=4),
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=90, redraw=True),
                                                    fromcurrent=True, transition=dict(duration=0))]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(mode="immediate",
                                                      transition=dict(duration=0),
                                                      frame=dict(duration=0, redraw=False))]),
                          ])],
        # timeline spans exactly the schematic width; step labels are hidden so only
        # the live "t = … s" readout shows (no crowded ruler)
        sliders=[dict(active=int(round(t / 5 * 50)), x=x0 + 0.26, len=xdom - 0.26,
                      y=1.06, ticklen=0, minorticklen=0,
                      font=dict(size=1, color="rgba(0,0,0,0)"),
                      currentvalue=dict(prefix="t = ", suffix=" s",
                                        font=dict(size=14, color=INK)),
                      pad=dict(t=2, b=4),
                      steps=[dict(method="animate", label=f"{ft:.1f}",
                                  args=[[f"{ft:.1f}"],
                                        dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for ft in fts])],
        plot_bgcolor=BLUEPRINT, paper_bgcolor=PAPER,
        font=dict(color=DIM),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.062,
                    font=dict(size=12, color=INK)),
        autosize=True, height=H,
        margin=dict(l=ML, r=MR, t=MT, b=MB), showlegend=not TRAJ_MODE)

    # schematic axes: true 1:1 aspect — never squashed, whatever the screen width
    fig.update_xaxes(range=[-0.85 * LR, 1.60 * LR], gridcolor=GRIDMAJ, zeroline=False,
                     ticksuffix=" m", constrain="domain",
                     minor=dict(showgrid=True, gridcolor=GRID), row=1, col=1)
    fig.update_yaxes(range=[-0.55 * LR, 1.45 * LR], gridcolor=GRIDMAJ, zeroline=False,
                     ticksuffix=" m", scaleanchor="x", scaleratio=1, constrain="domain",
                     minor=dict(showgrid=True, gridcolor=GRID), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=DATUM, line_width=1, row=1, col=1)

    # chart axes — square panels with fine minor grids for detail
    for r, cc in ((2, 1), (2, 2)):
        fig.update_xaxes(gridcolor=CH_GRID, row=r, col=cc,
                         minor=dict(showgrid=True, gridcolor=CH_GRIDMIN))
        fig.update_yaxes(gridcolor=CH_GRID, row=r, col=cc,
                         minor=dict(showgrid=True, gridcolor=CH_GRIDMIN))
    if TRAJ_MODE:
        for cc, xax in ((1, "x2"), (2, "x3")):
            fig.update_xaxes(title_text="X-position [m]", constrain="domain", row=2, col=cc)
            fig.update_yaxes(title_text="Y-position [m]", scaleanchor=xax, scaleratio=1,
                             constrain="domain", row=2, col=cc)
    else:
        # lock each panel to a square shape at any window size: pixels-per-unit on y
        # is tied to x so the domain shrinks instead of the panel stretching
        (kL, labL, _), (kR, labR, _) = TIME_MODES[charts]
        for cc, key, lab, xax in ((1, kL, labL, "x2"), (2, kR, labR, "x3")):
            ylo, yhi = yrange(key)
            fig.update_xaxes(title_text="time [s]", range=[0, 5], constrain="domain",
                             row=2, col=cc)
            fig.update_yaxes(title_text=lab, range=[ylo, yhi], constrain="domain",
                             scaleanchor=xax, scaleratio=5.0 / (yhi - ylo), row=2, col=cc)
    return fig


# ---------------------------------------------------------------- layout
left, right = st.columns([2.2, 1], gap="medium")

with left:
    st.subheader("Robot schematic & live charts", divider="orange")
    st.caption("θ measured from the horizontal · ▶ Play runs the arm **and** the charts in sync — "
               "drag the in-figure slider to scrub.")
    st.plotly_chart(build_figure(), width="stretch",
                    key=f"fig-{view}-{charts}",
                    config=dict(displayModeBar=False, responsive=True))

with right:
    st.subheader("Live state", divider="orange")
    st.markdown(
        f"<div style='font-family:monospace;font-size:13px'>"
        f"<div style='color:{S1}'>S1 · θ={s1['th']:.2f} rad · X={s1['x']:.2f} Y={s1['y']:.2f} m · "
        f"Q={fmt(s1['Q'])} Nm · P={fmt(s1['P'])} W</div>"
        f"<div style='color:{S2}'>S2 · θ={s2['th']:.2f} rad · X={s2['x']:.2f} Y={s2['y']:.2f} m · "
        f"Q={fmt(s2['Q'])} Nm · P={fmt(s2['P'])} W</div></div>",
        unsafe_allow_html=True)

    pp1 = float(np.max(np.abs(f1["P"])))
    pp2 = float(np.max(np.abs(f2["P"])))
    hi = 2 if pp2 >= pp1 else 1
    st.info(f"**Peak electrical power:** Scenario {hi} is higher — "
            f"{fmt(max(pp1, pp2))} W vs {fmt(min(pp1, pp2))} W. "
            f"The motor must be sized for the peak torque and peak power.")

    eq_lab = "Scenario 2" if both else view
    pe = PAR[2] if both else PAR[SCEN[0]]
    st.markdown(f"**Live equations — {eq_lab}** &nbsp;·&nbsp; t = {t:.2f} s")
    st.latex(r"Q = m\,l^{2}\ddot\theta + C\dot\theta + \tfrac{mgl}{2}\cos\theta")
    st.latex(rf"Q = \textcolor{{{COL_IN}}}{{{pe['m']:.0f}\cdot{pe['l']:.2f}^{{2}}\cdot{fs['dd']:.3f}}}"
             rf" + \textcolor{{{COL_DA}}}{{{pe['C']:.0f}\cdot{fs['d']:.2f}}}"
             rf" + \textcolor{{{COL_GR}}}{{\tfrac{{{pe['m']:.0f}\cdot 9.81\cdot{pe['l']:.2f}}}{{2}}\cos {fs['th']:.2f}}}")
    st.latex(rf"Q = \textcolor{{{COL_IN}}}{{{fs['inertia']:,.0f}}}"
             rf" + \textcolor{{{COL_DA}}}{{{fs['damp']:,.0f}}}"
             rf" + \textcolor{{{COL_GR}}}{{{fs['grav']:,.0f}}}"
             rf" = \mathbf{{{fs['Q']:,.0f}}}\ \mathrm{{Nm}}")
    st.latex(rf"P = Q\,\dot\theta = {fs['Q']:,.0f}\cdot{fs['d']:.2f}"
             rf" = \mathbf{{{fs['P']:,.0f}}}\ \mathrm{{W}}")
    st.caption("**Live:** every number above follows the **t** slider and the parameters — "
               "type a value next to any slider and it plugs straight into the formula. "
               "Colours match the torque terms below.")

    head = "both scenarios" if both else view
    st.markdown(f"**Torque breakdown — {head}** &nbsp;·&nbsp; at t = {t:.2f} s")

    def row(sw, label, hint, v1, v2=None):
        val = (f"<span style='color:{S1}'>S1 {fmt(v1)}</span> · "
               f"<span style='color:{S2}'>S2 {fmt(v2)}</span> Nm" if v2 is not None
               else f"<b style='color:{sw}'>{fmt(v1)} Nm</b>")
        return (f"<div style='display:flex;gap:8px;align-items:baseline;"
                f"font-family:monospace;font-size:13px;margin:3px 0'>"
                f"<span style='width:10px;height:10px;border-radius:2px;background:{sw};"
                f"flex:none;position:relative;top:1px'></span>"
                f"<span style='flex:1'>{label} <small style='color:{DIM}'>{hint}</small></span>"
                f"<span>{val}</span></div>")

    if both:
        rows = (row(COL_IN, "Inertia m·l²·θ̈", "to accelerate the arm", s1["inertia"], s2["inertia"])
                + row(COL_DA, "Damping C·θ̇", "to overcome friction", s1["damp"], s2["damp"])
                + row(COL_GR, "Gravity (m·g·l/2)·cosθ", "to hold it up", s1["grav"], s2["grav"]))
    else:
        rows = (row(COL_IN, "Inertia m·l²·θ̈", "to accelerate the arm", fs["inertia"])
                + row(COL_DA, "Damping C·θ̇", "to overcome friction", fs["damp"])
                + row(COL_GR, "Gravity (m·g·l/2)·cosθ", "to hold it up", fs["grav"]))
    st.markdown(rows, unsafe_allow_html=True)

    def total_box(label, v1, v2, unit):
        """Two-scenario total that never truncates (st.metric clips long values)."""
        return (f"<div style='background:{CARD_BG};border:1px solid {CARD_BORDER};"
                f"border-radius:8px;padding:9px 12px'>"
                f"<div style='font-size:12px;color:{DIM}'>{label}</div>"
                f"<div style='font-family:monospace;font-size:19px;font-weight:600;"
                f"white-space:nowrap'><span style='color:{S1}'>{v1}</span>"
                f" · <span style='color:{S2}'>{v2}</span>"
                f" <span style='font-size:13px;color:{DIM}'>{unit}</span></div>"
                f"<div style='font-size:11px;color:{DIM}'>S1 · S2</div></div>")

    def total_single(label, v, unit):
        return (f"<div style='background:{CARD_BG};border:1px solid {CARD_BORDER};"
                f"border-radius:8px;padding:9px 12px'>"
                f"<div style='font-size:12px;color:{DIM}'>{label}</div>"
                f"<div style='font-family:monospace;font-size:21px;font-weight:600;"
                f"white-space:nowrap;color:{INK}'>{v}"
                f" <span style='font-size:13px;color:{DIM}'>{unit}</span></div></div>")

    q1, q2 = st.columns(2)
    if both:
        q1.markdown(total_box("Total torque Q", fmt(s1["Q"]), fmt(s2["Q"]), "Nm"),
                    unsafe_allow_html=True)
        q2.markdown(total_box("Power P = Q·θ̇", fmt(s1["P"]), fmt(s2["P"]), "W"),
                    unsafe_allow_html=True)
    else:
        q1.markdown(total_single("Total torque Q", fmt(fs["Q"]), "Nm"),
                    unsafe_allow_html=True)
        q2.markdown(total_single("Power P = Q·θ̇", fmt(fs["P"]), "W"),
                    unsafe_allow_html=True)

    if show_annual:
        st.subheader(f"Annual — fleet of {R:.0f}", divider="orange")
        st.latex(rf"E_e=\int_0^{{5}}\!|P|\,dt"
                 rf" \;\Rightarrow\; E_{{e,1}}={E1 / 1000:.2f}\ \mathrm{{kJ}},"
                 rf"\;\; E_{{e,2}}={E2 / 1000:.2f}\ \mathrm{{kJ}}")
        a1, a2, a3 = st.columns(3)
        a1.metric("Energy / cycle", f"{E1 / 1000:.2f} kJ", f"S2: {E2 / 1000:.2f} kJ",
                  delta_color="off")
        a2.metric("Annual cost", f"£{fmt(c1)}", f"S2: £{fmt(c2)}", delta_color="off")
        a3.metric("Annual CO₂", f"{g1 / 1e6:.1f} t", f"S2: {g2 / 1e6:.1f} t", delta_color="off")
        st.caption("Big number = Scenario 1 · grey number = Scenario 2")

st.divider()
st.caption(f"θ₁(t) = {PAR[1]['a']:g}t² + {PAR[1]['b']:g}t · "
           f"θ₂(t) = {PAR[2]['a']:g}t³ + {PAR[2]['b']:g}t · 0 ≤ t ≤ 5 s — "
           "trapezoidal integration, 501 samples · course defaults: 0.02t² + 0.2t and "
           "0.01t³ + 0.05t")
