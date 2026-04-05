import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL Optimizer Comparison | NIT Delhi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e2e4ed; }
.main-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.75rem; font-weight: 600; color: #a5f3c4; letter-spacing: -0.02em; }
.sub-title { font-size: 0.82rem; color: #5c6370; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.3rem; }
.section-hdr { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #5c6370; text-transform: uppercase; letter-spacing: 0.12em; border-bottom: 1px solid #1e2130; padding-bottom: 0.35rem; margin: 1.6rem 0 0.9rem 0; }
.mcard { background: #13151d; border: 1px solid #1e2130; border-radius: 10px; padding: 1rem 1.2rem; }
.mlabel { font-size: 0.68rem; color: #5c6370; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'IBM Plex Mono', monospace; }
.mval { font-size: 1.65rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; line-height: 1.15; margin-top: 0.15rem; }
.msub { font-size: 0.72rem; color: #5c6370; font-family: 'IBM Plex Mono', monospace; margin-top: 0.1rem; }
.insight { background: #10120a; border-left: 3px solid #4ade80; padding: 0.7rem 0.9rem; border-radius: 0 8px 8px 0; font-size: 0.82rem; color: #9ca3af; line-height: 1.65; margin: 0.4rem 0; }
div[data-testid="stSidebar"] { background-color: #10121a; border-right: 1px solid #1e2130; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
OPT_COLORS = {"GWO": "#4ade80", "PSO": "#60a5fa", "ABC": "#fb923c"}
DRONE_COUNT = 10

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#10121a",
    font=dict(family="IBM Plex Mono, monospace", color="#9ca3af", size=11),
    margin=dict(l=10, r=10, t=36, b=10), hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2130", borderwidth=1),
)
AXIS_STYLE = dict(gridcolor="#191c28", linecolor="#1e2130", tickcolor="#1e2130", zeroline=False)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_defaults():
    paths = {
        "GWO": "gwo_federated_logs.xls",
        "PSO": "fed_pso_effnetb0_logs.xls",
        "ABC": "fed_logs_abc_optimizer.xls",
    }
    out = {}
    for opt, fname in paths.items():
        try:
            df = pd.read_csv(fname)
            out[opt] = standardize(df, opt)
        except Exception:
            out[opt] = None
    return out

def standardize(df, opt):
    df = df.copy()
    df.columns = df.columns.str.strip()
    renames = {
        "Global Accuracy (%)": "accuracy",
        "Global Acc":          "accuracy",
        "Total Latency (ms)":  "latency_ms",
        "Cloud Latency (ms)":  "cloud_latency_ms",
        "Average Cost":        "avg_cost",
        "Acceptance Rate A":   "accept_a",
        "Acceptance A":        "accept_a",
        "Acceptance Rate B":   "accept_b",
        "Acceptance B":        "accept_b",
        "Avg Fitness A":       "fitness_a",
        "Fitness A":           "fitness_a",
        "Avg Fitness B":       "fitness_b",
        "Fitness B":           "fitness_b",
        "Round":               "round",
    }
    df = df.rename(columns=renames)
    # rename drone loss cols to standard form
    drone_renames = {}
    for c in df.columns:
        if "Drone" in c and "Loss" in c:
            num = c.replace("Drone","").replace("Loss","").strip()
            drone_renames[c] = f"drone_{num}_loss"
    df = df.rename(columns=drone_renames)
    df["optimizer"] = opt
    return df

def load_uploaded(files):
    mapping = {}
    for f in files:
        name = f.name.lower()
        if "gwo" in name:   opt = "GWO"
        elif "pso" in name: opt = "PSO"
        elif "abc" in name: opt = "ABC"
        else:               opt = f.name.split(".")[0].upper()
        try:
            df = pd.read_csv(f)
            mapping[opt] = standardize(df, opt)
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
    return mapping

def avg_fitness(df):
    if "fitness_a" in df.columns and "fitness_b" in df.columns:
        return (df["fitness_a"] + df["fitness_b"]) / 2
    elif "fitness_a" in df.columns:
        return df["fitness_a"]
    return pd.Series([0]*len(df))

def drone_loss_cols(df):
    return sorted([c for c in df.columns if c.startswith("drone_") and c.endswith("_loss")])

# ─── Chart builders ───────────────────────────────────────────────────────────
def make_line(data, y_col, title, y_title, smooth, pct=False, invert=False):
    fig = go.Figure()
    for opt, df in data.items():
        if df is None or y_col not in df.columns:
            continue
        y = df[y_col].rolling(smooth, min_periods=1).mean()
        if invert:
            y = -y
        fmt = ".1f%%" if pct else ".2f"
        fig.add_trace(go.Scatter(
            x=df["round"], y=y, name=opt, mode="lines",
            line=dict(color=OPT_COLORS[opt], width=2.5),
            hovertemplate=f"<b>{opt}</b> Rd %{{x}} — {y_title}: %{{y:{fmt}}}<extra></extra>"
        ))
    fig.update_layout(**CHART_BASE,
        title=dict(text=title, font=dict(color="#a5f3c4", size=12)),
        xaxis=dict(title="Round", **AXIS_STYLE),
        yaxis=dict(title=y_title, **AXIS_STYLE),
    )
    return fig

def make_latency_bar(data, smooth):
    fig = go.Figure()
    for opt, df in data.items():
        if df is None or "latency_ms" not in df.columns:
            continue
        y = df["latency_ms"].rolling(smooth, min_periods=1).mean()
        fig.add_trace(go.Bar(
            x=df["round"], y=y, name=opt,
            marker_color=OPT_COLORS[opt], opacity=0.85,
            hovertemplate=f"<b>{opt}</b> Rd %{{x}} — %{{y:.0f}} ms<extra></extra>"
        ))
    fig.update_layout(**CHART_BASE,
        title=dict(text="Total latency per round (ms)", font=dict(color="#a5f3c4", size=12)),
        xaxis=dict(title="Round", **AXIS_STYLE),
        yaxis=dict(title="Latency (ms)", **AXIS_STYLE),
        barmode="group",
    )
    return fig

def make_cost(data, smooth):
    fig = go.Figure()
    for opt, df in data.items():
        if df is None or "avg_cost" not in df.columns:
            continue
        y = df["avg_cost"].rolling(smooth, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["round"], y=y, name=opt, mode="lines+markers",
            marker=dict(size=5, color=OPT_COLORS[opt]),
            line=dict(color=OPT_COLORS[opt], width=2, dash="dot"),
            hovertemplate=f"<b>{opt}</b> Rd %{{x}} — cost {{}:.4f}<extra></extra>"
        ))
    fig.update_layout(**CHART_BASE,
        title=dict(text="Average system cost per round", font=dict(color="#a5f3c4", size=12)),
        xaxis=dict(title="Round", **AXIS_STYLE),
        yaxis=dict(title="Cost", **AXIS_STYLE),
    )
    return fig

def make_fitness(data, smooth):
    fig = go.Figure()
    for opt, df in data.items():
        if df is None:
            continue
        y = avg_fitness(df).rolling(smooth, min_periods=1).mean()
        hex_c = OPT_COLORS[opt].lstrip("#")
        r,g,b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
        fig.add_trace(go.Scatter(
            x=df["round"], y=y, name=opt, mode="lines",
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.07)",
            line=dict(color=OPT_COLORS[opt], width=2.5),
            hovertemplate=f"<b>{opt}</b> Rd %{{x}} — fitness %{{y:.4f}}<extra></extra>"
        ))
    fig.update_layout(**CHART_BASE,
        title=dict(text="Average fitness score (Fog Broker A+B)", font=dict(color="#a5f3c4", size=12)),
        xaxis=dict(title="Round", **AXIS_STYLE),
        yaxis=dict(title="Avg Fitness", **AXIS_STYLE),
    )
    return fig

def make_drone_heatmap(df, opt):
    cols = drone_loss_cols(df)
    if not cols:
        return None
    z = df[cols].values.T
    labels = [c.replace("drone_","Drone ").replace("_loss","") for c in cols]
    fig = go.Figure(go.Heatmap(
        z=z, x=df["round"].tolist(), y=labels,
        colorscale=[[0,"#0d1f1a"],[0.5,OPT_COLORS[opt]],[1,"#fff"]],
        hovertemplate="Round %{x} — %{y}: loss %{z:.4f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#9ca3af", family="IBM Plex Mono", size=10))
    ))
    fig.update_layout(**CHART_BASE,
        title=dict(text=f"{opt} — per-drone loss heatmap", font=dict(color="#a5f3c4", size=12)),
        xaxis=dict(title="Round", **AXIS_STYLE, dtick=1),
        yaxis=dict(**AXIS_STYLE),
        margin=dict(l=70, r=10, t=36, b=10),
    )
    return fig

def make_acceptance(data, smooth):
    fig = go.Figure()
    for opt, df in data.items():
        if df is None or "accept_a" not in df.columns:
            continue
        for router, col, dash in [("Router A","accept_a","solid"), ("Router B","accept_b","dot")]:
            y = df[col].rolling(smooth, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df["round"], y=y, name=f"{opt} {router}",
                line=dict(color=OPT_COLORS[opt], width=2, dash=dash),
                hovertemplate=f"<b>{opt} {router}</b> Rd %{{x}} — %{{y:.2f}}<extra></extra>"
            ))
    fig.update_layout(**CHART_BASE,
        title=dict(text="Fog broker acceptance rate (Router A solid, B dotted)", font=dict(color="#a5f3c4", size=12)),
        xaxis=dict(title="Round", **AXIS_STYLE),
        yaxis=dict(title="Acceptance Rate", **AXIS_STYLE),
    )
    return fig

def make_radar(data):
    metrics = ["Accuracy", "Low Latency", "Low Cost", "Fitness"]
    fig = go.Figure()
    for opt, df in data.items():
        if df is None:
            continue
        acc  = df["accuracy"].mean() if "accuracy" in df.columns else 0
        lat  = df["latency_ms"].mean() if "latency_ms" in df.columns else 1
        cost = df["avg_cost"].mean() if "avg_cost" in df.columns else 1
        fit  = avg_fitness(df).mean()

        all_lats  = [d["latency_ms"].mean() for d in data.values() if d is not None and "latency_ms" in d.columns] or [1]
        all_costs = [d["avg_cost"].mean() for d in data.values() if d is not None and "avg_cost" in d.columns] or [1]
        all_accs  = [d["accuracy"].mean() for d in data.values() if d is not None and "accuracy" in d.columns] or [1]
        all_fits  = [avg_fitness(d).mean() for d in data.values() if d is not None]

        v = [
            (acc - min(all_accs)) / (max(all_accs) - min(all_accs) + 1e-9),
            1 - (lat - min(all_lats)) / (max(all_lats) - min(all_lats) + 1e-9),
            1 - (cost - min(all_costs)) / (max(all_costs) - min(all_costs) + 1e-9),
            (fit - min(all_fits)) / (max(all_fits) - min(all_fits) + 1e-9),
        ]
        v_closed = v + [v[0]]
        hex_c = OPT_COLORS[opt].lstrip("#")
        r,g,b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
        fig.add_trace(go.Scatterpolar(
            r=v_closed, theta=metrics+[metrics[0]], name=opt, fill="toself",
            line=dict(color=OPT_COLORS[opt], width=2),
            fillcolor=f"rgba({r},{g},{b},0.10)"
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="#10121a",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e2130", color="#5c6370", showticklabels=False),
            angularaxis=dict(gridcolor="#1e2130", color="#9ca3af", tickfont=dict(family="IBM Plex Mono", size=11))
        ),
        font=dict(family="IBM Plex Mono", color="#9ca3af", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30,r=30,t=40,b=30),
        title=dict(text="Overall performance radar (normalised)", font=dict(color="#a5f3c4", size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.68rem;color:#5c6370;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem'>⚙ Controls</div>", unsafe_allow_html=True)

    uploads = st.file_uploader(
        "Upload your XLS/CSV files (gwo, pso, abc)",
        type=["xls","csv"], accept_multiple_files=True, label_visibility="visible"
    )

    st.markdown("---")
    selected = st.multiselect("Optimizers", ["GWO","PSO","ABC"], default=["GWO","PSO","ABC"])
    smooth   = st.slider("Smoothing window", 1, 5, 1)
    drone_opt = st.selectbox("Drone heatmap for", ["GWO","PSO","ABC"])

    st.markdown("---")
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.68rem;color:#5c6370;line-height:1.8'>Sakshi Saraswat<br>NIT Delhi · 2025<br>GWECA, Ajmer<br>EfficientNet-B0 · FedAvg<br>10 drones · fog-cloud</div>", unsafe_allow_html=True)

# ─── Load data ────────────────────────────────────────────────────────────────
if uploads:
    raw = load_uploaded(uploads)
    using_sample = False
else:
    raw = load_defaults()
    using_sample = True

data = {k: v for k, v in raw.items() if k in selected and v is not None}

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🧠 Federated Learning — Optimizer Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>GWO · PSO · ABC &nbsp;|&nbsp; EfficientNet-B0 &nbsp;|&nbsp; Fog-Cloud Simulation &nbsp;|&nbsp; NIT Delhi 2025</div>", unsafe_allow_html=True)

if using_sample:
    st.info("✅ Loaded your real experimental data from the uploaded XLS files.", icon="📊")

# ─── Summary cards ────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Summary — averaged over all 10 rounds</div>", unsafe_allow_html=True)

cols = st.columns(len(data) if data else 1)
for i, (opt, df) in enumerate(data.items()):
    with cols[i]:
        acc  = df["accuracy"].mean() if "accuracy" in df.columns else 0
        lat  = df["latency_ms"].mean() if "latency_ms" in df.columns else 0
        cost = df["avg_cost"].mean() if "avg_cost" in df.columns else 0
        fit  = avg_fitness(df).mean()
        best_acc_round = int(df.loc[df["accuracy"].idxmax(), "round"]) if "accuracy" in df.columns else "—"
        c = OPT_COLORS[opt]
        st.markdown(f"""
        <div class='mcard' style='border-top:2.5px solid {c}'>
          <div class='mlabel'>{opt} optimizer</div>
          <div class='mval' style='color:{c}'>{acc:.1f}%</div>
          <div class='msub'>avg global accuracy</div>
          <div class='msub' style='margin-top:0.4rem'>⏱ {lat:,.0f} ms avg latency</div>
          <div class='msub'>💰 {cost:.4f} avg cost</div>
          <div class='msub'>📈 {fit:.4f} avg fitness</div>
          <div class='msub'>🏆 best at round {best_acc_round}</div>
        </div>""", unsafe_allow_html=True)

# ─── Row 1: Accuracy + Latency ───────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Accuracy & latency</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(make_line(data, "accuracy", "Global accuracy per round (%)", "Accuracy (%)", smooth), use_container_width=True)
with c2:
    st.plotly_chart(make_latency_bar(data, smooth), use_container_width=True)

# ─── Row 2: Fitness + Cost ────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Fitness & system cost</div>", unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(make_fitness(data, smooth), use_container_width=True)
with c4:
    st.plotly_chart(make_cost(data, smooth), use_container_width=True)

# ─── Row 3: Acceptance rate + Radar ──────────────────────────────────────────
st.markdown("<div class='section-hdr'>Fog broker acceptance & overall radar</div>", unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(make_acceptance(data, smooth), use_container_width=True)
with c6:
    st.plotly_chart(make_radar(data), use_container_width=True)

# ─── Per-drone loss heatmap ───────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Per-drone loss heatmap (10 drones × 10 rounds)</div>", unsafe_allow_html=True)
if drone_opt in data:
    hmap = make_drone_heatmap(data[drone_opt], drone_opt)
    if hmap:
        st.plotly_chart(hmap, use_container_width=True)
    else:
        st.info("No drone loss columns found in this file.")

# ─── Winner analysis ──────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Winner analysis</div>", unsafe_allow_html=True)
w1, w2 = st.columns([1, 1])

def winner(metric, higher=True):
    scores = {}
    for opt, df in data.items():
        if df is None: continue
        if metric == "fitness":
            scores[opt] = avg_fitness(df).mean()
        elif metric in df.columns:
            scores[opt] = df[metric].mean()
    if not scores: return "N/A", 0
    w = max(scores, key=scores.get) if higher else min(scores, key=scores.get)
    return w, scores[w]

with w1:
    for label, metric, higher, unit in [
        ("Highest accuracy",    "accuracy",    True,  "%"),
        ("Lowest latency",      "latency_ms",  False, " ms"),
        ("Lowest system cost",  "avg_cost",    False, ""),
        ("Best fitness score",  "fitness",     True,  ""),
    ]:
        w, val = winner(metric, higher)
        c = OPT_COLORS.get(w, "#fff")
        st.markdown(f"""<div class='insight'>
            <span style='font-size:0.72rem;color:#5c6370'>{label}</span><br>
            <span style='color:{c};font-weight:600;font-family:IBM Plex Mono'>{w}</span>
            <span style='color:#3d4452;font-size:0.75rem'> — {val:.2f}{unit}</span>
        </div>""", unsafe_allow_html=True)

with w2:
    st.markdown("""<div style='background:#10120a;border:1px solid #1e2130;border-radius:10px;padding:1rem 1.2rem;font-size:0.8rem;color:#9ca3af;line-height:1.9;font-family:IBM Plex Mono'>
    <span style='color:#a5f3c4;font-weight:600'>Research context</span><br>
    Model: EfficientNet-B0 (fine-tuned)<br>
    Architecture: 10 drones → Fog Broker → Cloud VM<br>
    Aggregation: Federated Averaging (FedAvg)<br>
    Rounds: 10 communication rounds<br>
    Local epochs: 5 per round<br>
    Simulation: fog-cloud with bandwidth noise<br>
    Dataset: 7-class image classification (Kaggle)<br>
    </div>""", unsafe_allow_html=True)

# ─── Raw data expander ────────────────────────────────────────────────────────
with st.expander("🗂 View raw data"):
    tab_labels = list(data.keys())
    if tab_labels:
        tabs = st.tabs(tab_labels)
        for tab, (opt, df) in zip(tabs, data.items()):
            with tab:
                st.dataframe(df.drop(columns=["optimizer"], errors="ignore"), use_container_width=True)
                st.download_button(
                    f"⬇ Download {opt} CSV",
                    df.to_csv(index=False),
                    f"{opt.lower()}_results.csv", "text/csv"
                )
