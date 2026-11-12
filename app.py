import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.ensemble import IsolationForest, RandomForestRegressor


def make_favicon():
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, 63, 63), radius=16, fill=(30, 64, 175, 255))
    d.polygon([(36, 8), (16, 36), (30, 36), (26, 56), (48, 26), (33, 26)], fill=(255, 255, 255, 255))
    return img


st.set_page_config(page_title="WattWise Campus", page_icon=make_favicon(), layout="wide")

PALETTE = {"Administration": "#2563EB", "Laboratories": "#F59E0B", "Library": "#10B981"}
INK, MUTED, LINE = "#0F172A", "#64748B", "#E2E8F0"

ICONS = {
    "bolt": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.3 17a10 10 0 1 1 17.4 0"/>',
    "peak": '<path d="M3 20l6-9 4 5 3-4 5 8z"/>',
    "moon": '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    "plug": '<path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 0 1-12 0z"/><path d="M12 18v4"/>',
    "alert": '<path d="M12 3 2 20h20z"/><path d="M12 10v5M12 18h.01"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "building": '<path d="M4 21V4h10v17M14 9h6v12M8 8h2M8 12h2M8 16h2M17 13h1M17 17h1M3 21h18"/>',
    "meter": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 14l3-3 3 2 4-5"/>',
}


def icon(name, size=20, color="currentColor"):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.9" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICONS[name]}</svg>'
    )


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
.stApp {background: #F6F8FC;}
.block-container {padding-top: 1.2rem; max-width: 1320px;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}

section[data-testid="stSidebar"] {background: #0B1220; border-right: 1px solid #1E293B;}
section[data-testid="stSidebar"] * {color: #CBD5E1;}
section[data-testid="stSidebar"] label p {color: #94A3B8 !important; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600;}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] {background: #131C2E; border-color: #26324A;}
section[data-testid="stSidebar"] .stDownloadButton button {
    background: #2563EB; color: #fff; border: 0; font-weight: 600; border-radius: 10px; padding: 10px;
}
section[data-testid="stSidebar"] .stDownloadButton button:hover {background: #1D4ED8;}
.brand {display: flex; align-items: center; gap: 12px; padding: 6px 0 18px;}
.brand .logo {width: 38px; height: 38px; border-radius: 11px; background: linear-gradient(135deg,#2563EB,#38BDF8); display: grid; place-items: center; color: #fff;}
.brand b {color: #F8FAFC; font-size: 1.12rem; letter-spacing: -.01em;}
.brand span {display: block; color: #64748B; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;}
.sb-note {background: #131C2E; border: 1px solid #26324A; border-radius: 12px; padding: 12px 14px; font-size: .8rem; color: #94A3B8; line-height: 1.5;}

.hero {
    position: relative; overflow: hidden; border-radius: 20px; padding: 34px 38px; margin-bottom: 20px; color: #fff;
    background:
      radial-gradient(900px 300px at 85% -20%, rgba(56,189,248,.45), transparent 60%),
      radial-gradient(600px 260px at 10% 120%, rgba(37,99,235,.55), transparent 60%),
      linear-gradient(120deg, #0B1220 0%, #12306B 100%);
    box-shadow: 0 18px 40px -18px rgba(18,48,107,.55);
}
.hero::after {content: ""; position: absolute; inset: 0; opacity: .08;
    background-image: linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px);
    background-size: 34px 34px;}
.hero > * {position: relative; z-index: 1;}
.pill {display: inline-flex; align-items: center; gap: 8px; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
    padding: 5px 13px; border-radius: 999px; font-size: .75rem; letter-spacing: .06em; text-transform: uppercase; font-weight: 600;}
.dot {width: 8px; height: 8px; border-radius: 50%; background: #34D399; box-shadow: 0 0 0 4px rgba(52,211,153,.25);}
.hero h1 {margin: 14px 0 4px; font-size: 2.5rem; font-weight: 800; letter-spacing: -.03em; color: #fff; padding: 0;}
.hero p {margin: 0; color: #BFD3FF; font-size: 1.05rem; max-width: 720px;}

.kpi {background: #fff; border: 1px solid #E5EAF3; border-radius: 16px; padding: 18px 20px 14px; box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.18);}
.kpi-top {display: flex; justify-content: space-between; align-items: center;}
.kpi .chip {width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center; background: var(--bg); color: var(--c);}
.kpi .l {color: #64748B; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600;}
.kpi .v {color: #0F172A; font-size: 1.95rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.2; margin-top: 10px;}
.kpi .v small {font-size: .95rem; font-weight: 600; color: #64748B; margin-left: 4px;}
.kpi .s {color: #64748B; font-size: .8rem; margin-top: 2px;}
.kpi svg.spark {display: block; margin-top: 10px; width: 100%; height: 34px;}

div[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid #E5EAF3; border-radius: 16px; padding: 10px 12px 2px; box-shadow: 0 1px 2px rgba(15,23,42,.04);}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #E5EAF3; border-radius: 14px; padding: 14px 18px;}
div[data-testid="stMetricLabel"] p {color: #64748B; font-weight: 600; text-transform: uppercase; font-size: .74rem; letter-spacing: .07em;}
div[data-testid="stMetricValue"] {font-weight: 800; color: #0F172A;}
div[data-testid="stDataFrame"] {border: 1px solid #E5EAF3; border-radius: 14px; overflow: hidden;}

.stTabs [data-baseweb="tab-list"] {gap: 4px; background: #E9EEF7; padding: 5px; border-radius: 12px; width: fit-content;}
.stTabs [data-baseweb="tab"] {height: 40px; padding: 0 22px; border-radius: 9px; background: transparent; font-weight: 600; color: #475569;}
.stTabs [aria-selected="true"] {background: #fff; color: #1E3A8A; box-shadow: 0 1px 3px rgba(15,23,42,.12);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

.sec {display: flex; align-items: baseline; gap: 12px; margin: 22px 0 10px;}
.sec h3 {margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A; padding: 0; letter-spacing: -.01em;}
.sec span {color: #64748B; font-size: .88rem;}

.rec {background: #fff; border: 1px solid #E5EAF3; border-left: 5px solid var(--c); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; display: flex; gap: 16px; align-items: flex-start; box-shadow: 0 1px 2px rgba(15,23,42,.04);}
.rec .chip {width: 40px; height: 40px; border-radius: 11px; display: grid; place-items: center; background: var(--bg); color: var(--c); flex: none;}
.rec b {color: #0F172A; font-size: 1rem;}
.rec p {margin: 3px 0 0; color: #64748B; font-size: .9rem;}
.tag {margin-left: auto; flex: none; font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; padding: 4px 10px; border-radius: 999px; background: var(--bg); color: var(--c);}
.notice {background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; border-radius: 12px; padding: 11px 16px; font-size: .85rem; margin-top: 12px;}
.foot {text-align: center; color: #94A3B8; font-size: .82rem; padding: 26px 0 8px;}
</style>
""",
    unsafe_allow_html=True,
)


def sparkline(values, color):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    span = (hi - lo) or 1.0
    xs = np.linspace(2, 198, len(v))
    ys = 32 - (v - lo) / span * 28
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f'<svg class="spark" viewBox="0 0 200 36" preserveAspectRatio="none">'
        f'<polygon points="2,34 {pts} 198,34" fill="{color}" opacity=".12"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/></svg>'
    )


def kpi(col, label, value, unit, sub, color, bg, ico, series):
    col.markdown(
        f'<div class="kpi" style="--c:{color};--bg:{bg}">'
        f'<div class="kpi-top"><div class="l">{label}</div>'
        f'<div class="chip">{icon(ico, 20)}</div></div>'
        f'<div class="v">{value}<small>{unit}</small></div>'
        f'<div class="s">{sub}</div>{sparkline(series, color)}</div>',
        unsafe_allow_html=True,
    )


def section(title, text=""):
    st.markdown(f'<div class="sec"><h3>{title}</h3><span>{text}</span></div>', unsafe_allow_html=True)


def style_fig(fig, title, ytitle="kWh", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK, family="Inter"), x=0.01),
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=-0.14, x=0, xanchor="left", title=None),
        hovermode="x unified",
        yaxis_title=ytitle,
        xaxis_title=None,
        font=dict(color=MUTED, family="Inter"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-09-01", periods=60 * 24, freq="h")
    buildings = {"Administration": 18, "Laboratories": 30, "Library": 22}
    rows = []
    for building, base in buildings.items():
        for date in dates:
            occupied = int(date.dayofweek < 5 and 8 <= date.hour < 18)
            consumption = base + occupied * base * 0.8
            consumption += 4 * np.sin(date.hour * np.pi / 12)
            consumption += rng.normal(0, 2)
            consumption = max(1, consumption)
            rows.append([date, building, occupied, round(consumption, 2)])
    data = pd.DataFrame(rows, columns=["timestamp", "building", "occupied", "consumption"])
    for building, event in [
        ("Laboratories", "2026-09-18 02:00"),
        ("Administration", "2026-10-04 01:00"),
        ("Library", "2026-10-21 03:00"),
    ]:
        event = pd.Timestamp(event)
        mask = (
            (data.building == building)
            & (data.timestamp >= event)
            & (data.timestamp < event + pd.Timedelta(hours=4))
        )
        data.loc[mask, "consumption"] *= 2.7
    return data


@st.cache_resource
def train_model(df):
    t = df.copy()
    t["hour"] = t.timestamp.dt.hour
    t["day"] = t.timestamp.dt.dayofweek
    codes = {b: i for i, b in enumerate(sorted(t.building.unique()))}
    t["building_code"] = t.building.map(codes)
    feats = ["hour", "day", "building_code", "occupied"]
    m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    m.fit(t[feats], t["consumption"])
    return m, codes, feats


df = generate_data()

with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="logo">{icon("bolt", 22)}</div>'
        f"<div><b>WattWise</b><span>Campus Energy AI</span></div></div>",
        unsafe_allow_html=True,
    )
    building = st.selectbox("Building", ["All"] + sorted(df.building.unique()))
    d_min, d_max = df.timestamp.dt.date.min(), df.timestamp.dt.date.max()
    period = st.date_input("Period", (d_min, d_max), min_value=d_min, max_value=d_max)
    sensitivity = st.slider(
        "Anomaly sensitivity", 0.5, 5.0, 1.5, 0.5, format="%.1f%%",
        help="Expected share of unusual hourly readings.",
    )
    st.write("")
    st.download_button(
        "Download demo dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="wattwise_synthetic_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.write("")
    st.markdown(
        '<div class="sb-note"><b style="color:#E2E8F0">Demo mode</b><br>'
        "All displayed energy data are synthetic, not real university measurements.</div>",
        unsafe_allow_html=True,
    )

data = df.copy()
if building != "All":
    data = data[data.building == building]
if isinstance(period, (tuple, list)) and len(period) == 2:
    data = data[(data.timestamp.dt.date >= period[0]) & (data.timestamp.dt.date <= period[1])]

if data.empty:
    st.error("No data for this selection.")
    st.stop()

st.markdown(
    """
<div class="hero">
  <span class="pill"><span class="dot"></span>Academic Prototype V1 &nbsp;·&nbsp; Synthetic data</span>
  <h1>WattWise Campus</h1>
  <p>AI-powered energy monitoring and optimization: track consumption, forecast demand and detect abnormal usage across university buildings.</p>
</div>
""",
    unsafe_allow_html=True,
)

total = data.consumption.sum()
average = data.consumption.mean()
peak_row = data.loc[data.consumption.idxmax()]
night = data[data.occupied == 0].consumption.mean()
day = data[data.occupied == 1].consumption.mean()
ratio = night / day * 100 if day and not np.isnan(day) else 0.0

dly = data.assign(date=data.timestamp.dt.date).groupby("date").consumption
s_total, s_avg, s_max = dly.sum(), dly.mean(), dly.max()
occ = data.assign(date=data.timestamp.dt.date).groupby(["date", "occupied"]).consumption.mean().unstack()
s_ratio = (occ[0] / occ[1]).dropna() if {0, 1}.issubset(occ.columns) else pd.Series(dtype=float)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Total consumption", f"{total:,.0f}", "kWh", f"{len(data):,} hourly readings", "#2563EB", "#EAF1FF", "bolt", s_total)
kpi(k2, "Average load", f"{average:.1f}", "kWh/h", "mean over selected hours", "#059669", "#E6F7F0", "gauge", s_avg)
kpi(k3, "Peak reading", f"{peak_row.consumption:.1f}", "kWh", f"{peak_row.building} · {peak_row.timestamp:%d %b, %H:%M}", "#D97706", "#FEF3DC", "peak", s_max)
kpi(k4, "Off-hours load", f"{ratio:.0f}", "%", "unoccupied vs occupied hours", "#DC2626", "#FDECEC", "moon", s_ratio)

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Consumption", "AI Forecast", "Anomalies", "Recommendations"])

with tab1:
    section("Energy Consumption Analysis", "Daily totals, hourly habits and building share")
    daily = data.assign(date=data.timestamp.dt.date).groupby(["date", "building"], as_index=False).consumption.sum()
    fig = px.line(daily, x="date", y="consumption", color="building", color_discrete_map=PALETTE)
    fig.update_traces(line=dict(width=2.6))
    st.plotly_chart(style_fig(fig, "Daily energy consumption", "kWh / day"), use_container_width=True)

    c1, c2 = st.columns([3, 2])
    profile = data.assign(hour=data.timestamp.dt.hour).groupby(["hour", "building"], as_index=False).consumption.mean()
    f_prof = px.line(profile, x="hour", y="consumption", color="building", color_discrete_map=PALETTE)
    f_prof.update_traces(line=dict(width=2.6))
    f_prof.update_xaxes(dtick=3)
    c1.plotly_chart(style_fig(f_prof, "Average hourly profile", "kWh", 380), use_container_width=True)

    share = data.groupby("building", as_index=False).consumption.sum()
    f_share = px.pie(share, names="building", values="consumption", hole=0.62, color="building", color_discrete_map=PALETTE)
    f_share.update_traces(textinfo="percent", textfont=dict(color="#fff", size=13), marker=dict(line=dict(color="#fff", width=3)))
    f_share.update_layout(
        title=dict(text="Share by building", font=dict(size=15, color=INK, family="Inter"), x=0.02),
        height=380, margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=MUTED),
        annotations=[dict(text=f"<b>{total / 1000:,.1f}</b><br>MWh", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=INK))],
    )
    c2.plotly_chart(f_share, use_container_width=True)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heat = (
        data.assign(hour=data.timestamp.dt.hour, dow=data.timestamp.dt.dayofweek)
        .pivot_table(index="dow", columns="hour", values="consumption", aggfunc="mean")
    )
    heat.index = [days[i] for i in heat.index]
    f_heat = px.imshow(
        heat, aspect="auto", color_continuous_scale=["#EAF1FF", "#7FA8FF", "#1E40AF", "#0B1B4D"],
        labels=dict(x="Hour of day", y="", color="kWh"),
    )
    f_heat.update_layout(
        title=dict(text="Load heatmap: weekday and hour", font=dict(size=15, color=INK, family="Inter"), x=0.01),
        height=360, margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=MUTED),
    )
    st.plotly_chart(f_heat, use_container_width=True)

with tab2:
    section("AI Consumption Forecast", "Random Forest model, next 24 hours")
    model, codes, feats = train_model(df)
    last = df.timestamp.max()
    future_idx = pd.date_range(last + pd.Timedelta(hours=1), periods=24, freq="h")
    names = data.building.unique()

    fc = pd.DataFrame(
        [
            [t, n, t.hour, t.dayofweek, codes[n], int(t.dayofweek < 5 and 8 <= t.hour < 18)]
            for n in names
            for t in future_idx
        ],
        columns=["timestamp", "building", "hour", "day", "building_code", "occupied"],
    )
    fc["predicted"] = model.predict(fc[feats])
    hist = df[(df.building.isin(names)) & (df.timestamp > last - pd.Timedelta(hours=48))]

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast total (24 h)", f"{fc.predicted.sum():,.0f} kWh")
    m2.metric("Forecast peak", f"{fc.predicted.max():.1f} kWh")
    m3.metric("Peak hour", f"{fc.loc[fc.predicted.idxmax(), 'timestamp']:%a %H:%M}")
    st.write("")

    fig2 = go.Figure()
    for n in names:
        h, f = hist[hist.building == n], fc[fc.building == n]
        fig2.add_trace(go.Scatter(x=h.timestamp, y=h.consumption, name=f"{n} · actual", line=dict(color=PALETTE[n], width=2.2), legendgroup=n))
        fig2.add_trace(go.Scatter(x=f.timestamp, y=f.predicted, name=f"{n} · forecast", line=dict(color=PALETTE[n], width=2.8, dash="dot"), legendgroup=n))
    fig2.add_vrect(
        x0=future_idx[0], x1=future_idx[-1], fillcolor="#DBEAFE", opacity=0.4, line_width=0,
        annotation_text="Next 24 h", annotation_position="top left",
        annotation_font=dict(color="#1E3A8A", size=12),
    )
    st.plotly_chart(style_fig(fig2, "Last 48 h and next 24 h forecast", height=440), use_container_width=True)
    st.markdown(
        '<div class="notice">Forecast is demonstrative and has not been validated against real campus measurements.</div>',
        unsafe_allow_html=True,
    )

with tab3:
    section("Anomaly Detection", "Isolation Forest on consumption, hour, weekday and building")
    ad = data.copy()
    ad["hour"] = ad.timestamp.dt.hour
    ad["day"] = ad.timestamp.dt.dayofweek
    ad["building_code"] = ad.building.map({b: i for i, b in enumerate(sorted(df.building.unique()))})
    det = IsolationForest(contamination=sensitivity / 100, random_state=42)
    ad["result"] = det.fit_predict(ad[["consumption", "hour", "day", "building_code"]])
    anomalies = ad[ad.result == -1]

    a1, a2, a3 = st.columns(3)
    a1.metric("Detected unusual events", len(anomalies))
    a2.metric("Worst reading", f"{anomalies.consumption.max():.1f} kWh" if len(anomalies) else "n/a")
    a3.metric("Share of readings", f"{len(anomalies) / len(ad) * 100:.1f}%")
    st.write("")

    fig3 = go.Figure()
    for n in ad.building.unique():
        s = ad[ad.building == n]
        fig3.add_trace(go.Scatter(x=s.timestamp, y=s.consumption, name=n, mode="lines", line=dict(color=PALETTE[n], width=1)))
    fig3.add_trace(go.Scatter(
        x=anomalies.timestamp, y=anomalies.consumption, name="Anomaly", mode="markers",
        marker=dict(color="#DC2626", size=8, symbol="circle-open", line=dict(width=2, color="#DC2626")),
    ))
    st.plotly_chart(style_fig(fig3, "Hourly consumption with detected anomalies", height=420), use_container_width=True)

    section("Top 15 unusual readings")
    table = anomalies[["timestamp", "building", "consumption"]].sort_values("consumption", ascending=False).head(15)
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="DD MMM YYYY, HH:mm"),
            "building": "Building",
            "consumption": st.column_config.ProgressColumn(
                "Consumption (kWh)", format="%.1f", min_value=0, max_value=float(max(df.consumption.max(), 1))
            ),
        },
    )

with tab4:
    section("Smart Energy Recommendations", "Prioritised from the current selection")
    recs = [
        ("plug", "Check equipment operating during unoccupied hours", f"Off-hours load reaches {ratio:.0f}% of the occupied-hours load.", "High", "#DC2626", "#FDECEC"),
        ("alert", "Investigate unusually high night consumption", f"{len(anomalies)} unusual events detected in the selected period.", "High", "#DC2626", "#FDECEC"),
        ("clock", "Schedule shutdown of non-essential equipment", "Automate lights, HVAC and lab equipment outside 08:00 to 18:00.", "Medium", "#D97706", "#FEF3DC"),
        ("building", "Compare energy profiles between campus buildings", "Use the hourly profile and heatmap to spot outlier buildings.", "Medium", "#2563EB", "#EAF1FF"),
        ("meter", "Collect real meter readings to improve AI predictions", "Real data are required to validate forecasts before deployment.", "Next step", "#059669", "#E6F7F0"),
    ]
    for ico, title, text, tag, c, bg in recs:
        st.markdown(
            f'<div class="rec" style="--c:{c};--bg:{bg}"><div class="chip">{icon(ico, 21)}</div>'
            f'<div><b>{title}</b><p>{text}</p></div><span class="tag">{tag}</span></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="foot">WattWise Campus &nbsp;|&nbsp; Academic Prototype V1 &nbsp;|&nbsp; Université de Saïda — Proposed Pilot</div>',
    unsafe_allow_html=True,
)