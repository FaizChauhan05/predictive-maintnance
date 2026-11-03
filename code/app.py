import os
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from dotenv import load_dotenv
from datetime import datetime, timezone
from llm_report import generate_report, generate_classification_report
from pathlib import Path
import cwru

DATA = Path(__file__).resolve().parent.parent / "data"

load_dotenv()
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### About this dashboard")
    st.markdown("""
This platform analyzes two types of industrial machines and estimates
their current health using sensor data and machine learning.

**Turbofan Engine Module**
- **Estimated Remaining Life** — How many operating cycles the engine
  likely has left before maintenance is needed. Higher is healthier.
- **Fleet Median** — A reference value representing the middle
  remaining life across the benchmark engine fleet.
- **Actual Remaining Life (Benchmark)** — The reference answer from
  the test dataset, used for model evaluation. The model never sees
  this value during prediction.

**Bearing Vibration Module**
- Analyzes vibration signals to detect bearing faults.
- Classifies each bearing as **Normal**, **Inner Race Fault**,
  **Ball Fault**, or **Outer Race Fault**.
- **Model Confidence** — How consistently the model identified
  the condition across all signal segments.

**Feature Contribution Charts**
Each sensor reading is scored by how much it influenced the result:
- **Blue bars** → Pushed toward a healthier estimate or the detected condition.
- **Pink bars** → Pushed toward a shorter life estimate or another condition.

**Maintenance Interpretation**
An AI reads the prediction and key contributing factors, then writes
a short advisory note in plain English. It is not a substitute for
professional engineering judgment.
""")

# ---------- THEME / CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Geist', sans-serif;
    background-color: #131315;
    color: #e5e1e4;
}
.stApp { background-color: #131315; }
h1, h2, h3 { font-family: 'Geist', sans-serif; letter-spacing: -0.02em; }

.pm-card {
    background: #1c1b1d;
    border: 1px solid #2a2a2c;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.pm-card-title {
    font-size: 13px;
    font-weight: 500;
    color: #e5e1e4;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}
.pm-metric {
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #4edea3;
    line-height: 1.1;
}
.pm-metric.muted { color: #88929b; }
.pm-unit {
    font-size: 14px;
    color: #bec8d2;
    margin-left: 6px;
    font-family: 'Geist', sans-serif;
}
.pm-sub {
    font-size: 11px;
    color: #88929b;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}
.pm-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    background: #2a2a2c;
    color: #bec8d2;
}
.pm-badge.green { background: rgba(78,222,163,0.12); color: #4edea3; }
.pm-badge.blue  { background: rgba(137,206,255,0.12); color: #89ceff; }
.pm-badge.amber { background: rgba(255,185,95,0.12); color: #ffb95f; }
.pm-badge.red   { background: rgba(255,180,171,0.12); color: #ffb4ab; }
.pm-health {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(78,222,163,0.12);
    color: #4edea3;
}
.pm-health .dot { width: 7px; height: 7px; border-radius: 50%; background: #4edea3; }
.pm-report {
    background: #201f22;
    border-radius: 10px;
    padding: 22px 24px;
    font-size: 14.5px;
    line-height: 1.7;
    color: #e5e1e4;
    text-align: justify;
}
.pm-report code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: #2a2a2c;
    padding: 2px 6px;
    border-radius: 4px;
    color: #ffb4ab;
}
.pm-provenance {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #71717a;
    letter-spacing: 0.04em;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #2a2a2c;
}
.pm-bar {
    width: 100%;
    height: 8px;
    background: #353437;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 10px;
}
.pm-bar-fill { height: 100%; border-radius: 999px; }
.pm-header-title {
    font-size: 18px; font-weight: 600; letter-spacing: -0.02em;
    color: #e5e1e4; display: flex; align-items: center; gap: 10px;
}
.pm-header-title .dot { width: 8px; height: 8px; border-radius: 50%; background: #4edea3; }
.pm-header-sub { font-size: 12px; color: #bec8d2; margin-top: 4px; }

.pm-sr-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}
.pm-sr-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #88929b;
    margin-bottom: 6px;
}
.pm-sr-pos { color: #89ceff; font-size: 13.5px; line-height: 1.5; }
.pm-sr-neg { color: #ffb4ab; font-size: 13.5px; line-height: 1.5; }
.pm-sr-action {
    background: #2a2a2c;
    border-radius: 8px;
    padding: 12px 16px;
}
.pm-card-explain {
    font-size: 12.5px;
    color: #88929b;
    line-height: 1.6;
    margin-top: 8px;
}
.pm-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #88929b;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2a2c;
}

/* Tabs (Streamlit's st.tabs) restyled to look like the mockup's top bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0e0e10;
    border-bottom: 1px solid #2a2a2c;
    padding: 0 24px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Geist', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #bec8d2;
    background: transparent;
    border-radius: 0;
    padding: 10px 16px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #89ceff !important;
    border-bottom: 2px solid #89ceff !important;
    background: #1c1b1d !important;
}
</style>
""", unsafe_allow_html=True)


# ---------- HELPER: Readable turbofan feature names ----------
def _readable_feature_name(raw_name, sensor_labels):
    """Convert raw feature name like 's11_delta' to a human-readable label."""
    parts = raw_name.split("_", 1)
    sensor_code = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    description = sensor_labels.get(sensor_code, sensor_code)
    if suffix == "delta":
        suffix_label = "Change"
    elif suffix == "std":
        suffix_label = "Variability"
    else:
        suffix_label = suffix
    return f"{description} \u00b7 {suffix_label}  [{raw_name}]"


# ---------- BEARING FEATURE LABELS ----------
BEARING_FEATURE_LABELS = {
    "mean": "Signal Average",
    "std": "Signal Variability",
    "rms": "Signal Intensity (RMS)",
    "peak": "Maximum Vibration Peak",
    "ptp": "Peak-to-Peak Range",
    "kurtosis": "Sharpness of Vibration Spikes",
    "skew": "Vibration Asymmetry",
    "crest": "Crest Factor",
    "shape": "Shape Factor",
    "impulse": "Impulse Factor",
    "band0_energy": "Low Freq. Energy (0\u2013500 Hz)",
    "band1_energy": "Mid-Low Freq. Energy (500\u20131500 Hz)",
    "band2_energy": "Mid Freq. Energy (1500\u20133000 Hz)",
    "band3_energy": "Mid-High Freq. Energy (3000\u20134500 Hz)",
    "band4_energy": "High Freq. Energy (4500\u20136000 Hz)",
    "spec_centroid": "Spectral Center of Mass",
    "spec_spread": "Spectral Spread",
}


# ---------- TABS ----------
tab_turbofan, tab_bearing = st.tabs(["\u2708  Turbofan Engine", "\u2699  Bearing Vibration"])


# ===========================================================================
# TAB 1 \u2014 TURBOFAN
# ===========================================================================
with tab_turbofan:
    COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    USEFUL = [f"s{i}" for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
    WINDOW = 10
    BASELINE = 63.0

    SENSOR_LABELS = {
        "s2": "LPC Outlet Temp.",
        "s3": "HPC Outlet Temp.",
        "s4": "LPT Coolant Temp.",
        "s7": "HPC Outlet Temp. (2)",
        "s8": "Physical Fan Speed",
        "s9": "Physical Core Speed",
        "s11": "Static Core Press.",
        "s12": "Fan Exit Temp.",
        "s13": "Corrected Fan Speed",
        "s14": "Corrected Core Speed",
        "s15": "Bypass Duct Press.",
        "s17": "Engine Press. Ratio",
        "s20": "HPT Coolant Flow",
        "s21": "Coolant Bleed Flow",
    }

    @st.cache_data
    def load_data():
        train = pd.read_csv(DATA / "train_FD001.txt", sep=r"\s+", header=None, names=COLS)
        m = train.groupby("unit")["cycle"].transform("max")
        train["RUL"] = (m - train["cycle"]).clip(upper=125)
        test = pd.read_csv(DATA / "test_FD001.txt", sep=r"\s+", header=None, names=COLS)
        rul_true = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
        return train, test, rul_true

    def features_rowwise(df, window=WINDOW):
        out = []
        for unit, g in df.groupby("unit"):
            g = g.sort_values("cycle").reset_index(drop=True)
            for i in range(len(g)):
                if i < window - 1:
                    continue
                tail = g.iloc[i - window + 1 : i + 1]
                row = {"unit": unit, "cycle": g.loc[i, "cycle"]}
                if "RUL" in g.columns:
                    row["RUL"] = g.loc[i, "RUL"]
                for c in USEFUL:
                    v = tail[c].values
                    row[f"{c}_std"]   = v.std()
                    row[f"{c}_delta"] = v[-1] - v[0]
                out.append(row)
        return pd.DataFrame(out)

    @st.cache_resource
    def train_model():
        train, _, _ = load_data()
        ftr = features_rowwise(train)
        low  = ftr[ftr["RUL"] <= 80]
        high = ftr[ftr["RUL"] >  80].sample(frac=0.3, random_state=42)
        ftr  = pd.concat([low, high]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        X = ftr.drop(columns=["unit", "cycle", "RUL"])
        y = ftr["RUL"].values
        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0,
            random_state=42,
        )
        model.fit(X, y)
        return model, shap.TreeExplainer(model)

    def features_for_engine(test_df, engine_id):
        g = test_df[test_df["unit"] == engine_id].sort_values("cycle").reset_index(drop=True)
        if len(g) < WINDOW:
            return None, None
        tail = g.tail(WINDOW)
        row = {}
        for c in USEFUL:
            v = tail[c].values
            row[f"{c}_std"]   = v.std()
            row[f"{c}_delta"] = v[-1] - v[0]
        return pd.DataFrame([row]), int(g["cycle"].iloc[-1])

    @st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
    def cached_report(engine_id, predicted_rul, baseline, top_features_tuple):
        return generate_report(
            engine_id=engine_id,
            predicted_rul=predicted_rul,
            baseline=baseline,
            top_features=list(top_features_tuple),
        )

    # ---- HEADER ----
    head_l, head_r = st.columns([3, 2])
    with head_l:
        st.markdown(
            '<div class="pm-header-title"><span class="dot"></span>'
            'Predictive Maintenance AI Platform</div>'
            '<div class="pm-header-sub">Turbofan engine health \u2014 estimated remaining life, contributing factors, and maintenance interpretation</div>',
            unsafe_allow_html=True,
        )
    with head_r:
        c1, c2 = st.columns([1, 1])
        with c1:
            engine_id = st.selectbox("Select Engine", list(range(1, 101)), index=46, label_visibility="collapsed")
        with c2:
            st.button("Generate AI Maintenance Report", use_container_width=True)

    st.markdown("---")

    # ---- LOAD MODEL ----
    with st.spinner("Loading engine analysis model \u2014 this may take a moment on the first run..."):
        model, explainer = train_model()
        _, test, rul_true = load_data()

    x_one, last_cycle = features_for_engine(test, engine_id)
    if x_one is None:
        st.error("Not enough cycles for this engine.")
        st.stop()

    pred = float(model.predict(x_one)[0])
    sv = explainer.shap_values(x_one)[0]
    pairs = sorted(zip(x_one.columns, sv), key=lambda t: abs(t[1]), reverse=True)[:10]
    true_rul = int(rul_true.iloc[engine_id - 1]["RUL"])

    ratio = max(0.0, min(1.0, pred / 125.0))
    if pred > 80:    bar_color = "#4edea3"; status = "Healthy"
    elif pred > 30:  bar_color = "#ffb95f"; status = "Caution"
    else:            bar_color = "#ffb4ab"; status = "Critical"

    left, right = st.columns([5, 7])

    # ---- LEFT COLUMN: METRIC CARDS ----
    with left:
        # Engine identity card
        st.markdown(f"""
        <div class="pm-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div class="pm-sub" style="margin:0 0 4px 0;">Turbofan Engine</div>
              <div style="font-size:26px;font-weight:600;letter-spacing:-0.02em;">Engine {engine_id}</div>
              <div class="pm-sub" style="margin-top:4px;">Operating Cycle #{last_cycle}</div>
            </div>
            <span class="pm-health"><span class="dot"></span>OPERATIONAL</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Estimated remaining life card
        if pred > 80:
            status_badge_class = "green"
            status_badge_text = f"Healthy \u2014 above 80 cycles"
        elif pred > 30:
            status_badge_class = "amber"
            status_badge_text = f"Caution \u2014 30 to 80 cycles"
        else:
            status_badge_class = "red"
            status_badge_text = f"Critical \u2014 below 30 cycles"

        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-card-title">Estimated Remaining Life
            <span class="pm-badge {status_badge_class}" style="margin-left:auto;">
              {status_badge_text}
            </span>
          </div>
          <div style="display:flex;align-items:baseline;">
            <span class="pm-metric" style="color:{bar_color};">{pred:.0f}</span>
            <span class="pm-unit">cycles</span>
          </div>
          <div class="pm-bar">
            <div class="pm-bar-fill" style="width:{ratio*100:.1f}%;background:{bar_color};"></div>
          </div>
          <div class="pm-sub" style="display:flex;justify-content:space-between;">
            <span>Critical (below 30)</span><span>Caution (30\u201380)</span><span>Healthy (above 80)</span>
          </div>
          <div class="pm-card-explain">Remaining useful life is the estimated number of operating cycles before maintenance is needed. One cycle represents a complete engine start-to-shutdown sequence. The scale ranges from 0 (end of life) to 125 (maximum).</div>
        </div>
        """, unsafe_allow_html=True)

        # Fleet median card
        delta = pred - BASELINE
        if delta > 0:
            fleet_badge_class = "green"
            fleet_badge_text = f"This engine: {abs(delta):.0f} cycles above fleet"
        elif delta < 0:
            fleet_badge_class = "amber"
            fleet_badge_text = f"This engine: {abs(delta):.0f} cycles below fleet"
        else:
            fleet_badge_class = "blue"
            fleet_badge_text = "This engine: at fleet median"

        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-card-title">Fleet Median Remaining Life
            <span class="pm-badge {fleet_badge_class}" style="margin-left:auto;">{fleet_badge_text}</span>
          </div>
          <div><span class="pm-metric" style="color:#e5e1e4;">{BASELINE:.0f}</span><span class="pm-unit">cycles</span></div>
          <div class="pm-card-explain">The fleet median is a reference value representing the middle predicted remaining life across the benchmark engine fleet. It helps gauge whether this engine is in better or worse condition relative to the fleet.</div>
        </div>
        """, unsafe_allow_html=True)

        # Actual benchmark card
        prediction_error = pred - true_rul
        if prediction_error > 0:
            err_color = "#89ceff"
            err_label = "(overestimated)"
        elif prediction_error < 0:
            err_color = "#ffb95f"
            err_label = "(underestimated)"
        else:
            err_color = "#4edea3"
            err_label = "(exact match)"

        st.markdown(f"""
        <div class="pm-card" style="background:#2a2a2c;">
          <div class="pm-card-title" style="color:#bec8d2;">Actual Remaining Life (Benchmark)
            <span class="pm-badge" style="margin-left:auto;">Benchmark</span>
          </div>
          <div><span class="pm-metric muted">{true_rul}</span><span class="pm-unit" style="color:#88929b;">cycles</span></div>
          <div style="margin-top:8px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{err_color};">
              Prediction Error: {prediction_error:+.0f} cycles {err_label}
            </span>
          </div>
          <div class="pm-card-explain">This is the reference answer from the test dataset, used to evaluate model accuracy. The model did not have access to this value when making its prediction. The prediction error shows how far the estimate was from this benchmark.</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- RIGHT COLUMN: SHAP CHART ----
    with right:
        st.markdown(
            '<div class="pm-card-title" style="font-size:20px;font-weight:600;">'
            'What influenced the estimated remaining life?</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="pm-card-explain" style="margin-bottom:12px;">'
            'Each bar shows how a sensor reading contributed to the model\u2019s estimated remaining life. '
            'Positive values (blue) increased the estimate, while negative values (pink) decreased it. '
            'The model identifies patterns in sensor data \u2014 these are correlations, not confirmed causes of wear.</div>',
            unsafe_allow_html=True,
        )

        names = [_readable_feature_name(p[0], SENSOR_LABELS) for p in pairs][::-1]
        vals  = [p[1] for p in pairs][::-1]
        colors = ["#0EA5E9" if v > 0 else "#ffb4ab" for v in vals]

        fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#1c1b1d")
        ax.set_facecolor("#1c1b1d")
        bars = ax.barh(names, vals, color=colors, height=0.62)
        ax.axvline(0, color="#3e4850", linewidth=1.2, linestyle="--")
        ax.tick_params(colors="#bec8d2", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_xlabel("Contribution to estimated remaining life (in cycles)",
                      color="#bec8d2", fontsize=11, labelpad=10)
        for bar, v in zip(bars, vals):
            x = bar.get_width()
            ax.text(x + (0.3 if v > 0 else -0.3), bar.get_y() + bar.get_height()/2,
                    f"{v:+.1f}", va="center",
                    ha="left" if v > 0 else "right",
                    color="#0EA5E9" if v > 0 else "#ffb4ab",
                    fontsize=9, fontfamily="monospace")
        fig.subplots_adjust(left=0.42)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Legend
        st.markdown(
            '<div style="display:flex;gap:24px;justify-content:center;margin-top:8px;">'
            '<div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#bec8d2;">'
            '<span style="width:12px;height:12px;background:#0EA5E9;border-radius:3px;display:inline-block;"></span>'
            'Increased remaining life estimate</div>'
            '<div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#bec8d2;">'
            '<span style="width:12px;height:12px;background:#ffb4ab;border-radius:3px;display:inline-block;"></span>'
            'Decreased remaining life estimate</div></div>',
            unsafe_allow_html=True,
        )

    # ---- MAINTENANCE INTERPRETATION ----
    st.markdown("", unsafe_allow_html=True)
    top8 = tuple((f, round(float(v), 2)) for f, v in pairs[:8])

    with st.spinner("Generating maintenance interpretation..."):
        report = cached_report(
            engine_id=int(engine_id),
            predicted_rul=round(pred, 1),
            baseline=BASELINE,
            top_features_tuple=top8,
        )

    if isinstance(report, dict):
        verdict        = report.get("verdict", "UNKNOWN")
        urgency        = report.get("urgency", "Routine")
        summary        = report.get("summary", "\u2014")
        drivers_pos    = report.get("drivers_positive", "\u2014")
        drivers_neg    = report.get("drivers_negative", "\u2014")
        recommendation = report.get("recommendation", "\u2014")
    else:
        verdict, urgency = "UNKNOWN", "Routine"
        summary, drivers_pos, drivers_neg, recommendation = str(report), "\u2014", "\u2014", "\u2014"

    verdict_colors = {
        "OPTIMAL":  ("#4edea3", "rgba(78,222,163,0.12)"),
        "CAUTION":  ("#ffb95f", "rgba(255,185,95,0.12)"),
        "CRITICAL": ("#ffb4ab", "rgba(255,180,171,0.12)"),
        "UNKNOWN":  ("#88929b", "rgba(136,146,155,0.12)"),
    }
    vc, vbg = verdict_colors.get(verdict, verdict_colors["UNKNOWN"])

    st.markdown(f"""
    <div class="pm-card">
      <div class="pm-card-title" style="font-size:18px;font-weight:600;">
        Maintenance Interpretation
        <span class="pm-badge" style="margin-left:auto;background:{vbg};color:{vc};">
          {verdict} \u00b7 {urgency}
        </span>
      </div>
      <div class="pm-section-label">Summary</div>
      <div style="font-size:15px;line-height:1.7;color:#e5e1e4;margin-bottom:18px;">{summary}</div>
      <div class="pm-section-label">What influenced the prediction</div>
      <div class="pm-sr-grid">
        <div><div class="pm-sr-label">Increased the estimate</div><div class="pm-sr-pos">{drivers_pos}</div></div>
        <div><div class="pm-sr-label">Decreased the estimate</div><div class="pm-sr-neg">{drivers_neg}</div></div>
      </div>
      <div class="pm-section-label">Recommended next step</div>
      <div class="pm-sr-action">
        <div style="font-size:14px;color:#e5e1e4;">{recommendation}</div>
      </div>
      <div style="font-size:11px;color:#71717a;margin-top:14px;font-style:italic;">This interpretation is generated by AI based on model outputs. It is advisory only and should not replace professional engineering judgment.</div>
    </div>
    """, unsafe_allow_html=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f'<div class="pm-provenance">Analyzed: {now} \u00b7 Engine life estimation model \u00b7 Feature contribution analysis</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# TAB 2 \u2014 BEARING (CWRU)
# ===========================================================================
with tab_bearing:
    BEARING_FILES = {
        "Inner race fault":  "1797_IR_7_DE12.npz",
        "Ball fault":        "1797_B_7_DE12.npz",
        "Outer race fault":  "1797_OR@6_7_DE12.npz",
        "Normal baseline":   "1797_Normal.npz",
    }

    @st.cache_resource
    def load_bearing_model():
        try:
            bundle = cwru.load_artifacts("models")
            return bundle["model"], bundle["feature_names"], bundle["class_names"]
        except FileNotFoundError:
            return None, None, None

    @st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
    def cached_bearing_report(bearing_id, predicted_class, confidence, top_features_tuple):
        return generate_classification_report(
            bearing_id=bearing_id,
            predicted_class=predicted_class,
            confidence=confidence,
            top_features=list(top_features_tuple),
        )

    # ---- HEADER ----
    head_l, head_r = st.columns([3, 2])
    with head_l:
        st.markdown(
            '<div class="pm-header-title"><span class="dot"></span>'
            'Predictive Maintenance AI Platform</div>'
            '<div class="pm-header-sub">Bearing vibration analysis \u2014 fault classification, contributing factors, and maintenance interpretation</div>',
            unsafe_allow_html=True,
        )
    with head_r:
        c1, c2 = st.columns([1, 1])
        with c1:
            bearing_label = st.selectbox(
                "Signal", list(BEARING_FILES.keys()), index=0,
                label_visibility="collapsed",
            )
        with c2:
            st.button("Generate AI Maintenance Report", key="bearing_gen", use_container_width=True)

    st.markdown("---")

    model_b, feature_names_b, class_names_b = load_bearing_model()
    if model_b is None:
        st.warning(
            "Bearing model not trained yet. Run `python code/train_cwru.py` once, "
            "then reload this page."
        )
        st.stop()

    mat_path = os.path.join(cwru.CWRU_DIR, BEARING_FILES[bearing_label])

    with st.spinner("Analyzing vibration signal..."):
        result = cwru.predict_file(model_b, feature_names_b, mat_path)

    majority_idx   = result["majority_class"]
    predicted_name = class_names_b[majority_idx]
    confidence     = result["confidence"]
    X_rep          = result["representative_X"]
    mean_proba     = result["mean_proba"]

    explainer_b = shap.TreeExplainer(model_b)
    shap_out = explainer_b.shap_values(X_rep)

    if isinstance(shap_out, list):
        sv_rep = shap_out[majority_idx][0]
    elif shap_out.ndim == 3:
        sv_rep = shap_out[0, :, majority_idx]
    else:
        sv_rep = shap_out[0]

    pairs_b = sorted(zip(feature_names_b, sv_rep), key=lambda t: abs(t[1]), reverse=True)[:10]

    status_map = {
        "Normal":     ("#4edea3", "OPTIMAL"),
        "Inner Race": ("#ffb95f", "CAUTION"),
        "Ball":       ("#ffb4ab", "CRITICAL"),
        "Outer Race": ("#ffb4ab", "CRITICAL"),
    }
    bar_color_b, status_b = status_map.get(predicted_name, ("#88929b", "UNKNOWN"))

    left, right = st.columns([5, 7])

    # ---- LEFT COLUMN ----
    with left:
        # Bearing identity card
        st.markdown(f"""
        <div class="pm-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div class="pm-sub" style="margin:0 0 4px 0;">Bearing Vibration Signal</div>
              <div style="font-size:26px;font-weight:600;letter-spacing:-0.02em;">
                {bearing_label.split('(')[0].strip()}
              </div>
              <div class="pm-sub" style="margin-top:4px;">{result['n_windows']} signal segments analyzed</div>
            </div>
            <span class="pm-health"
                  style="background:{'rgba(78,222,163,0.12)' if status_b=='OPTIMAL' else ('rgba(255,185,95,0.12)' if status_b=='CAUTION' else 'rgba(255,180,171,0.12)')};
                         color:{bar_color_b};">
              <span class="dot" style="background:{bar_color_b};"></span>{status_b}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Detected bearing condition card
        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-card-title">
            <span style="display:inline-flex;align-items:center;gap:6px;">
              <span style="color:{bar_color_b};">\u25c9</span> Detected Bearing Condition
            </span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:600;
                      letter-spacing:-0.02em;color:{bar_color_b};margin-bottom:8px;">
            {predicted_name}
          </div>
          <div class="pm-bar">
            <div class="pm-bar-fill" style="width:{confidence*100:.1f}%;background:{bar_color_b};"></div>
          </div>
          <div class="pm-sub">Model Confidence: {confidence:.1%} &nbsp; (avg across {result['n_windows']} segments)</div>
          <div class="pm-card-explain">The model analyzed vibration patterns to classify the bearing into one of four conditions. Model confidence indicates how consistently the model identified this condition across all signal segments. This is not a guaranteed probability of real-world failure.</div>
        </div>
        """, unsafe_allow_html=True)

        # Condition meanings expander
        with st.expander("What do these conditions mean?"):
            st.markdown("""
- **Normal** \u2014 No detectable fault in the bearing. Vibration patterns are within expected ranges.
- **Inner Race Fault** \u2014 A defect on the inner ring of the bearing. Often caused by fatigue, contamination, or improper lubrication.
- **Ball Fault** \u2014 A defect on one or more rolling elements (balls). May indicate early-stage wear or material fatigue.
- **Outer Race Fault** \u2014 A defect on the outer ring of the bearing. One of the most common bearing failure modes.
            """)

        # Model scores bar chart (replaces plain text probabilities)
        st.markdown("""
        <div class="pm-card-title" style="margin-top:16px;margin-bottom:8px;">
          <span style="display:inline-flex;align-items:center;gap:6px;">
            <span style="color:#88929b;">\u25a4</span> Model Scores for Each Condition
          </span>
        </div>
        """, unsafe_allow_html=True)

        fig_prob, ax_prob = plt.subplots(figsize=(5, 2.5), facecolor="#1c1b1d")
        ax_prob.set_facecolor("#1c1b1d")
        prob_colors = []
        for i, name in enumerate(class_names_b):
            if name == predicted_name:
                prob_colors.append(bar_color_b)
            else:
                prob_colors.append("#3e4850")
        y_positions = list(range(len(class_names_b)))
        prob_vals = [p * 100 for p in mean_proba]
        bars_prob = ax_prob.barh(y_positions, prob_vals, color=prob_colors, height=0.55)
        ax_prob.set_yticks(y_positions)
        ax_prob.set_yticklabels(class_names_b)
        ax_prob.tick_params(colors="#bec8d2", labelsize=10)
        ax_prob.set_xlabel("Score (%)", color="#bec8d2", fontsize=10, labelpad=8)
        ax_prob.set_xlim(0, 105)
        for spine in ax_prob.spines.values():
            spine.set_visible(False)
        ax_prob.grid(False)
        for bar_p, pv in zip(bars_prob, prob_vals):
            ax_prob.text(
                bar_p.get_width() + 1.5,
                bar_p.get_y() + bar_p.get_height() / 2,
                f"{pv:.1f}%",
                va="center", ha="left",
                color="#e5e1e4", fontsize=9, fontfamily="monospace",
            )
        fig_prob.tight_layout()
        st.pyplot(fig_prob, use_container_width=True)

        st.markdown(
            '<div class="pm-card-explain">These scores show the model\u2019s confidence for each bearing '
            'condition. The highest score determines the detected condition. '
            'Scores across all conditions sum to 100%.</div>',
            unsafe_allow_html=True,
        )

    # ---- RIGHT COLUMN: SHAP CHART ----
    with right:
        st.markdown(
            '<div class="pm-card-title" style="font-size:20px;font-weight:600;margin-bottom:4px;">'
            'Which vibration patterns influenced the result?</div>',
            unsafe_allow_html=True,
        )

        # Legend
        st.markdown(
            '<div style="display:flex;gap:24px;padding:8px 12px;background:#1c1b1d;'
            'border-radius:6px;margin-bottom:10px;">'
            '<div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#bec8d2;">'
            '<span style="width:12px;height:12px;background:#ffb4ab;border-radius:3px;"></span>'
            'Argued against this condition</div>'
            '<div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#bec8d2;">'
            '<span style="width:12px;height:12px;background:#0EA5E9;border-radius:3px;"></span>'
            f'Supported the {predicted_name} classification</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        raw_names = [p[0] for p in pairs_b][::-1]
        readable_names = [f"{BEARING_FEATURE_LABELS.get(n, n)}  [{n}]" for n in raw_names]
        vals  = [float(p[1]) for p in pairs_b][::-1]
        colors = ["#0EA5E9" if v > 0 else "#ffb4ab" for v in vals]

        fig, ax = plt.subplots(figsize=(8, 5.2), facecolor="#1c1b1d")
        ax.set_facecolor("#1c1b1d")
        bars = ax.barh(readable_names, vals, color=colors, height=0.62)
        ax.axvline(0, color="#88929b", linewidth=1.0, linestyle=(0, (3, 3)))
        ax.tick_params(colors="#bec8d2", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_xlabel(
            f"Contribution to {predicted_name} classification",
            color="#bec8d2", fontsize=11, labelpad=10,
        )
        for bar, v in zip(bars, vals):
            x = bar.get_width()
            ax.text(
                x + (0.003 if v > 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}",
                va="center",
                ha="left" if v > 0 else "right",
                color="#0EA5E9" if v > 0 else "#ffb4ab",
                fontsize=9,
                fontfamily="monospace",
            )
        fig.subplots_adjust(left=0.42)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown(
            '<div class="pm-card-explain" style="text-align:center;">'
            'This chart identifies which vibration patterns the model used for classification. '
            'These are statistical patterns in the data, not confirmed causes of bearing damage. '
            'Positive contributions supported the detected condition, while negative contributions suggested other conditions.</div>',
            unsafe_allow_html=True,
        )

        # Expandable glossary
        with st.expander("Understanding vibration features"):
            st.markdown("""
**Signal Intensity (RMS)** \u2014 Root Mean Square: the overall energy level of the vibration signal. Higher values often indicate more severe vibration.

**Sharpness of Vibration Spikes (Kurtosis)** \u2014 Measures how "peaky" the signal is. High kurtosis suggests sharp, impulsive vibrations typical of bearing defects.

**Frequency Band Energy** \u2014 The vibration signal is split into frequency ranges. Different fault types tend to concentrate energy in specific bands.

**Spectral Center of Mass** \u2014 The "average" frequency weighted by energy. Shifts in this value can indicate changes in the vibration pattern.
            """)

    # ---- MAINTENANCE INTERPRETATION ----
    st.markdown("", unsafe_allow_html=True)
    top8_b = tuple((f, round(float(v), 4)) for f, v in pairs_b[:8])

    with st.spinner("Generating maintenance interpretation..."):
        report_b = cached_bearing_report(
            bearing_id=bearing_label,
            predicted_class=predicted_name,
            confidence=round(confidence, 3),
            top_features_tuple=top8_b,
        )

    if isinstance(report_b, dict):
        v_b = report_b.get("verdict", "UNKNOWN")
        u_b = report_b.get("urgency", "Routine")
        s_b = report_b.get("summary", "\u2014")
        dp_b = report_b.get("drivers_positive", "\u2014")
        dn_b = report_b.get("drivers_negative", "\u2014")
        r_b = report_b.get("recommendation", "\u2014")
    else:
        v_b, u_b = "UNKNOWN", "Routine"
        s_b, dp_b, dn_b, r_b = str(report_b), "\u2014", "\u2014", "\u2014"

    vc_b, vbg_b = verdict_colors.get(v_b, verdict_colors["UNKNOWN"])

    st.markdown(f"""
    <div class="pm-card">
      <div class="pm-card-title" style="font-size:18px;font-weight:600;">
        Maintenance Interpretation
        <span class="pm-badge" style="margin-left:auto;background:{vbg_b};color:{vc_b};">
          {v_b} \u00b7 {u_b}
        </span>
      </div>
      <div class="pm-section-label">Summary</div>
      <div style="font-size:15px;line-height:1.7;color:#e5e1e4;margin-bottom:18px;">{s_b}</div>
      <div class="pm-section-label">What influenced the classification</div>
      <div class="pm-sr-grid">
        <div>
          <div class="pm-sr-label">Supported this classification</div>
          <div class="pm-sr-pos">{dp_b}</div>
        </div>
        <div>
          <div class="pm-sr-label">Argued against this classification</div>
          <div class="pm-sr-neg">{dn_b}</div>
        </div>
      </div>
      <div class="pm-section-label">Recommended next step</div>
      <div class="pm-sr-action">
        <div style="font-size:14px;color:#e5e1e4;">{r_b}</div>
      </div>
      <div style="font-size:11px;color:#71717a;margin-top:14px;font-style:italic;">This interpretation is generated by AI based on model outputs. It is advisory only and should not replace professional engineering judgment.</div>
    </div>
    """, unsafe_allow_html=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f'<div class="pm-provenance">Analyzed: {now} \u00b7 Bearing classification model \u00b7 Feature contribution analysis</div>',
        unsafe_allow_html=True,
    )
