"""ENG3032 - Robot Arm Simulation Lab.

Virtual laboratory for 1/2/3-link planar serial arms:
  * Joint control  — pose the arm with sliders (forward kinematics)
  * Click target   — FABRIK inverse kinematics chases the point you click
  * Motion playback — course scenarios (1-link, exact dynamics) or
                      Wave / Weld-seam motions (multi-link)

Run with:  streamlit run welding_robot_lab.py   (this page appears in the sidebar)
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ENG3032 · Simulation Lab", page_icon="🕹️", layout="wide")

# cap and centre the content on ultra-wide displays — no dead space on the right
st.markdown("<style>.block-container{max-width:1750px;margin:0 auto;}</style>",
            unsafe_allow_html=True)

G = 9.81

# ---------------------------------------------------------------- theme palette
_theme = getattr(st.context, "theme", None)
DARK = getattr(_theme, "type", "light") == "dark"

if DARK:
    INK, DIM = "#e9eff6", "#9aabbd"
    PAPER, BLUEPRINT = "#101828", "#0e1a2f"
    GRID, GRIDMAJ, DATUM = "#1c2d4b", "#293d63", "#46587a"
    ANN_BG = "rgba(13,20,34,0.92)"
    TARGETCOL, TIPCOL, TRACECOL = "#ff6e57", "#f2a93b", "#f2a93b"
    LINKCOL = ["#6fa9ff", "#46c47c", "#f2a93b"]
else:
    INK, DIM = "#16222e", "#5d6b78"
    PAPER, BLUEPRINT = "white", "#eef4fb"
    GRID, GRIDMAJ, DATUM = "#dae6f5", "#b3c9e6", "#9aa8b4"
    ANN_BG = "rgba(255,255,255,0.92)"
    TARGETCOL, TIPCOL, TRACECOL = "#c8331f", "#e8920a", "#e8920a"
    LINKCOL = ["#1f5fc8", "#1f7a3d", "#e8920a"]


# ---------------------------------------------------------------- kinematics
def joints(L, A):
    """Joint positions [n+1, 2] from link lengths and absolute angles."""
    p = np.zeros((len(L) + 1, 2))
    for i, (li, ai) in enumerate(zip(L, A)):
        p[i + 1] = p[i] + li * np.array([np.cos(ai), np.sin(ai)])
    return p


def fabrik(L, target, A0, iters=10):
    """FABRIK inverse kinematics — returns absolute link angles."""
    L = np.asarray(L, float)
    n = len(L)
    tgt = np.asarray(target, float)
    if np.hypot(*tgt) >= L.sum():                  # out of reach: point at it
        return np.full(n, np.arctan2(tgt[1], tgt[0]))
    p = joints(L, A0)
    for _ in range(iters):
        p[n] = tgt
        for i in range(n - 1, -1, -1):             # backward pass
            v = p[i] - p[i + 1]
            d = np.hypot(*v) or 1e-9
            p[i] = p[i + 1] + v / d * L[i]
        p[0] = 0.0
        for i in range(n):                         # forward pass
            v = p[i + 1] - p[i]
            d = np.hypot(*v) or 1e-9
            p[i + 1] = p[i] + v / d * L[i]
    return np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))


def course_theta(which, tt):
    """Exact course motions: th, th', th'' at time tt."""
    if which == 1:
        return 0.02 * tt**2 + 0.2 * tt, 0.04 * tt + 0.2, 0.04
    return 0.01 * tt**3 + 0.05 * tt, 0.03 * tt**2 + 0.05, 0.06 * tt


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🕹️ Lab controls")
    nlink = st.selectbox("Robot", [1, 2, 3], index=1,
                         format_func=lambda n: {1: "1-link — welding arm (course model)",
                                                2: "2-link planar arm",
                                                3: "3-link planar arm"}[n])
    mode = st.radio("Mode", ["Joint control", "Click target — IK", "Motion playback"])
    st.divider()
    DEF_L = [1.2] if nlink == 1 else [1.0, 0.8, 0.5]
    L = [st.slider(f"Link length l{i + 1} [m]", 0.4, 1.5, DEF_L[i], 0.01) for i in range(nlink)]
    reach = float(sum(L))
    A_sliders = None
    if mode == "Joint control":
        st.divider()
        A_sliders = np.array([st.slider(f"Joint angle θ{i + 1} [rad]", -1.57, 1.57,
                                        [0.35, 0.15, 0.10][i], 0.01) for i in range(nlink)])
    m_kg = C_d = None
    if nlink == 1:
        st.divider()
        m_kg = st.slider("Mass m [kg]", 50, 700, 250, 1)
        C_d = st.slider("Damping C [Nm·s/rad]", 0, 300, 100, 1)
    preset = None
    if mode == "Motion playback":
        st.divider()
        if nlink == 1:
            preset = st.selectbox("Motion", ["Scenario 1 — θ=0.02t²+0.2t",
                                             "Scenario 2 — θ=0.01t³+0.05t"])
        else:
            preset = st.selectbox("Motion", ["Wave — joints out of phase",
                                             "Weld seam — IK follows the seam"])

# ---------------------------------------------------------------- click-target state
if "ik_target" not in st.session_state:
    st.session_state.ik_target = (0.8, 0.6)


def _clicked_point(state):
    try:
        pts = state["selection"]["points"]
        if pts:
            return float(pts[0]["x"]), float(pts[0]["y"])
    except Exception:
        pass
    return None


if mode == "Click target — IK":
    prev = st.session_state.get("stage")
    hit = _clicked_point(prev) if prev is not None else None
    if hit:
        st.session_state.ik_target = (round(hit[0], 3), round(hit[1], 3))

# ---------------------------------------------------------------- current pose
if mode == "Joint control":
    A_now = A_sliders
elif mode == "Click target — IK":
    A_now = fabrik(L, st.session_state.ik_target, np.full(nlink, 0.3))
else:
    A_now = None        # playback: pose comes from the animation frames

SEAM = mode == "Motion playback" and preset is not None and preset.startswith("Weld seam")


# ---------------------------------------------------------------- stage drawing
def sc(**kw):
    kw.setdefault("hoverinfo", "skip")
    kw.setdefault("showlegend", False)
    return go.Scatter(**kw)


def poly(xs, ys, fill, line="#1f262d", w=1.5):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return sc(x=np.append(xs, xs[0]), y=np.append(ys, ys[0]), mode="lines",
              fill="toself", fillcolor=fill, line=dict(color=line, width=w))


def stage_static():
    """Blueprint base, hub, dashed reach circle, datum, optional seam, click grid."""
    b = 0.10 * reach
    aa = np.linspace(0, 2 * np.pi, 72)
    T = [
        poly([-b, b, b, -b], [0, 0, -1.6 * b, -1.6 * b], "#b9a02d", "#5d5414"),
        poly(0.55 * b * np.cos(np.linspace(0, np.pi, 24)),
             0.55 * b * np.sin(np.linspace(0, np.pi, 24)), "#c9ced4", "#5d5414"),
        sc(x=reach * np.cos(aa), y=reach * np.sin(aa), mode="lines",
           line=dict(color=GRIDMAJ, width=1, dash="dash")),
        sc(x=[-0.3 * reach, 1.18 * reach], y=[0, 0], mode="lines",
           line=dict(color=DATUM, width=1, dash="dash")),
    ]
    if SEAM:
        sy, x0, x1 = 0.55 * reach, -0.10 * reach, 0.75 * reach
        T.append(sc(x=[x0, x1], y=[sy, sy], mode="lines",
                    line=dict(color=TARGETCOL, width=3, dash="dash")))
        T.append(sc(x=[x1], y=[sy], mode="text", text=["  weld seam"],
                    textposition="middle right",
                    textfont=dict(color=TARGETCOL, size=11, family="IBM Plex Mono, monospace")))
    if mode == "Click target — IK":
        gx, gy = np.meshgrid(np.linspace(-reach, reach, 41),
                             np.linspace(-0.25 * reach, 1.1 * reach, 28))
        T.append(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers",
                            marker=dict(size=14, color="rgba(0,0,0,0.001)"),
                            hovertemplate="X=%{x:.2f} Y=%{y:.2f} m<extra>click to set target</extra>",
                            showlegend=False))
    return T


def arm_traces(A, trace_pts=None, spark_seed=0):
    """Coloured links, joint pins, tip dot, optional tip trace and sparks."""
    p = joints(L, A)
    T = []
    if trace_pts is not None and len(trace_pts) > 1:
        T.append(sc(x=trace_pts[:, 0], y=trace_pts[:, 1], mode="lines", opacity=0.85,
                    line=dict(color=TRACECOL, width=2)))
    for i in range(nlink):
        T.append(sc(x=[p[i, 0], p[i + 1, 0]], y=[p[i, 1], p[i + 1, 1]], mode="lines",
                    line=dict(color=LINKCOL[i], width=11 - 2 * i)))
    T.append(sc(x=p[:nlink, 0], y=p[:nlink, 1], mode="markers",
                marker=dict(size=[11] + [9] * (nlink - 1),
                            color=["#c9ced4"] + [TARGETCOL] * (nlink - 1),
                            line=dict(color="#16222e", width=1.5))))
    T.append(sc(x=[p[nlink, 0]], y=[p[nlink, 1]], mode="markers",
                marker=dict(size=11, color=TIPCOL, line=dict(color="#16222e", width=1.5))))
    rng = np.random.default_rng(spark_seed + 1)
    nsp = 6 if spark_seed else 0
    T.append(sc(x=p[nlink, 0] + rng.normal(0, 0.03 * reach, nsp),
                y=p[nlink, 1] + rng.normal(-0.01 * reach, 0.03 * reach, nsp),
                mode="markers", opacity=0.9,
                marker=dict(symbol="star", color="#ff9d1f", size=rng.uniform(4, 9, nsp))))
    if mode == "Click target — IK":
        tx, ty = st.session_state.ik_target
        T.append(sc(x=[tx], y=[ty], mode="markers",
                    marker=dict(symbol="x-thin", size=16, color=TARGETCOL,
                                line=dict(color=TARGETCOL, width=3))))
    return T


def tip_callout(A, extra=""):
    p = joints(L, A)
    x, y = p[nlink]
    return [dict(x=x, y=y, xref="x", yref="y", showarrow=True,
                 text=f"X={x:.2f}  Y={y:.2f} m" + extra,
                 font=dict(color=INK, size=12, family="IBM Plex Mono, monospace"),
                 bgcolor=ANN_BG, bordercolor=TIPCOL, borderwidth=1.2,
                 borderpad=3, arrowcolor=TIPCOL, arrowwidth=1.2, ax=60, ay=-46)]


STAGE_W, STAGE_PH, STAGE_MT, STAGE_MB = 1100, 640, 90, 12
STAGE_H = STAGE_PH + STAGE_MT + STAGE_MB


def stage_layout(fig):
    fig.update_layout(plot_bgcolor=BLUEPRINT, paper_bgcolor=PAPER, font=dict(color=DIM),
                      margin=dict(l=10, r=10, t=STAGE_MT, b=STAGE_MB),
                      autosize=True, height=STAGE_H, showlegend=False, dragmode=False)
    fig.update_xaxes(range=[-1.15 * reach, 1.30 * reach], gridcolor=GRIDMAJ,
                     zeroline=False, ticksuffix=" m", constrain="domain",
                     minor=dict(showgrid=True, gridcolor=GRID))
    fig.update_yaxes(range=[-0.30 * reach, 1.15 * reach], gridcolor=GRIDMAJ,
                     zeroline=False, ticksuffix=" m", scaleanchor="x", scaleratio=1,
                     constrain="domain", minor=dict(showgrid=True, gridcolor=GRID))
    return fig


# ---------------------------------------------------------------- playback poses
def playback():
    """List of (t, angles, kin-or-None) plus the time vector."""
    if nlink == 1:
        which = 1 if preset.startswith("Scenario 1") else 2
        ts = np.linspace(0, 5, 51)
        return [(tt, np.array([course_theta(which, tt)[0]]), course_theta(which, tt))
                for tt in ts], ts
    if preset.startswith("Wave"):
        ts = np.linspace(0, 7, 71)
        return [(tt, 0.55 + 0.45 * np.sin(0.9 * tt + 0.95 * np.arange(nlink)), None)
                for tt in ts], ts
    ts = np.linspace(0, 4 * np.pi, 64)                       # weld seam (one full sweep)
    sy, x0, x1 = 0.55 * reach, -0.10 * reach, 0.75 * reach
    a_cur, out = np.full(nlink, 0.6), []
    for tt in ts:
        u = (np.sin(0.5 * tt - np.pi / 2) + 1) / 2
        a_cur = fabrik(L, (x0 + (x1 - x0) * u, sy), a_cur)
        out.append((tt, a_cur.copy(), None))
    return out, ts


def playback_figure():
    poses, ts = playback()
    tips = np.array([joints(L, A)[nlink] for _, A, _ in poses])

    def extra(kin):
        if kin is None:
            return ""
        th, d, dd = kin
        Q = m_kg * L[0]**2 * dd + C_d * d + m_kg * G * L[0] / 2 * np.cos(th)
        return f"<br>Q={Q:,.0f} Nm · P={Q * d:,.0f} W"

    def frame_data(k):
        return stage_static() + arm_traces(poses[k][1], tips[:k + 1],
                                           spark_seed=int(poses[k][0] * 100) + 1)

    fig = go.Figure(data=frame_data(0))
    idx = list(range(len(fig.data)))           # every trace is rebuilt per frame
    fig.frames = [go.Frame(data=frame_data(k), traces=idx, name=f"{k}",
                           layout=dict(annotations=tip_callout(poses[k][1],
                                                               extra(poses[k][2]))))
                  for k in range(len(poses))]
    fig.update_layout(
        annotations=tip_callout(poses[0][1], extra(poses[0][2])),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.075, showactive=False,
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=90, redraw=True),
                                                    fromcurrent=True, transition=dict(duration=0))]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(mode="immediate",
                                                      transition=dict(duration=0),
                                                      frame=dict(duration=0, redraw=False))]),
                          ])],
        # step labels hidden — only the live "t = … s" readout shows
        sliders=[dict(active=0, x=0.32, len=0.68, y=1.072, ticklen=0, minorticklen=0,
                      font=dict(size=1, color="rgba(0,0,0,0)"),
                      currentvalue=dict(prefix="t = ", suffix=" s",
                                        font=dict(size=14, color=INK)),
                      pad=dict(t=2, b=4),
                      steps=[dict(method="animate", label=f"{poses[k][0]:.1f}",
                                  args=[[f"{k}"], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True))])
                             for k in range(len(poses))])])
    return stage_layout(fig)


# ---------------------------------------------------------------- page
st.title("Robot Arm Simulation Lab")
st.caption("ENG3032 · virtual laboratory — serial planar arms · FABRIK inverse kinematics · "
           "1-link dynamics identical to the course model")

left, right = st.columns([2.3, 1], gap="medium")

with left:
    tag = {"Joint control": "drag the sliders to pose the arm",
           "Click target — IK": "click anywhere on the stage — FABRIK solves the angles",
           "Motion playback": "press ▶ Play and watch the tip trace its path"}[mode]
    st.subheader("Simulation stage", divider="orange")
    st.caption(tag)
    if mode == "Motion playback":
        st.plotly_chart(playback_figure(), width="stretch",
                        key=f"stage-play-{nlink}-{preset}",
                        config=dict(displayModeBar=False, responsive=True))
    else:
        fig = go.Figure(data=stage_static() + arm_traces(A_now))
        fig.update_layout(annotations=tip_callout(A_now))
        stage_layout(fig)
        if mode == "Click target — IK":
            st.plotly_chart(fig, key="stage", on_select="rerun",
                            selection_mode=("points",),
                            config=dict(displayModeBar=False, responsive=True),
                            width="stretch")
        else:
            st.plotly_chart(fig, key=f"stage-joint-{nlink}", width="stretch",
                            config=dict(displayModeBar=False, responsive=True))

with right:
    st.subheader("Readout", divider="orange")
    if A_now is not None:
        p = joints(L, A_now)
        for i, a in enumerate(A_now):
            st.latex(rf"\theta_{{{i + 1}}} = {a:.2f}\,\mathrm{{rad}} = {np.degrees(a):.0f}^\circ")
        if nlink == 1:
            st.latex(rf"X = l\cos\theta = {p[1, 0]:.2f}\ \mathrm{{m}}")
            st.latex(rf"Y = l\sin\theta = {p[1, 1]:.2f}\ \mathrm{{m}}")
        else:
            st.latex(rf"X = \textstyle\sum_i l_i\cos\theta_i = {p[nlink, 0]:.2f}\ \mathrm{{m}}")
            st.latex(rf"Y = \textstyle\sum_i l_i\sin\theta_i = {p[nlink, 1]:.2f}\ \mathrm{{m}}")
    st.latex(rf"\mathrm{{reach}} = \textstyle\sum_i l_i = {reach:.2f}\ \mathrm{{m}}")

    if mode == "Click target — IK":
        tx, ty = st.session_state.ik_target
        st.latex(rf"\mathbf{{x}}_{{\mathrm{{target}}}} = ({tx:.2f},\ {ty:.2f})\ \mathrm{{m}}")
        st.caption("The ⨯ marks the target. FABRIK iterates 10× per solve; outside the "
                   "dashed reach circle the arm simply stretches towards the target.")

    if nlink == 1:
        st.latex(r"Q = m\,l^{2}\ddot\theta + C\dot\theta + \tfrac{mgl}{2}\cos\theta")
        st.latex(r"P = Q\,\dot\theta")
        if A_now is not None:
            Qh = m_kg * G * L[0] / 2 * float(np.cos(A_now[0]))
            st.latex(rf"Q_{{\mathrm{{hold}}}} = \tfrac{{mgl}}{{2}}\cos\theta = {Qh:,.0f}\ \mathrm{{Nm}}")
            st.caption("Static pose (θ̇ = θ̈ = 0): only the gravity term acts. Switch to "
                       "Motion playback to see the full course dynamics live on the stage.")
        else:
            st.caption("During playback the callout on the stage shows the exact course "
                       "dynamics Q and P at every instant.")

    if mode == "Motion playback" and nlink > 1:
        st.caption("**Wave** = forward kinematics from prescribed joint angles. "
                   "**Weld seam** = inverse kinematics chasing a moving point on the seam — "
                   "the orange trace is the tip path.")

st.divider()
st.caption("FABRIK inverse kinematics · single-link dynamics identical to the course model "
           "Q = m·l²·θ̈ + C·θ̇ + (m·g·l/2)·cosθ · P = Q·θ̇")
