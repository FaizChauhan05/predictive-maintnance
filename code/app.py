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
**What it does:**
Predictive maintenance platform with two machine modules sharing one
explainability and AI-report pipeline. Picks a unit from a test set,
estimates its condition, explains which sensor readings drove that
estimate, and asks an AI to write a short maintenance note in plain
English.

**Turbofan (C-MAPSS FD001)**
- **Predicted RUL** — the model's best guess of how many cycles the
  engine has left. Higher is healthier.
- **Population Baseline** — the average predicted RUL across the
  training fleet (63 cycles).
- **True RUL (Answer Key)** — ground truth from the dataset, shown
  for validation only. The model never sees it.

**Bearing (CWRU)**
- Vibration signal is windowed; 15 time + frequency domain features
  are extracted per window.
- XGBoost classifier predicts one of: Normal / Inner Race / Ball /
  Outer Race.
- **Confidence** = average predicted probability across all windows.

**The chart on the right — "Why this prediction?"**
Every reading gets scored by how much it pushed the prediction up or
down:
- **Blue bars (right)** = pushes toward *more life* / *current class*.
- **Pink bars (left)** = pushes toward *less life* / *other classes*.

**The AI Maintenance Report**
Gemini reads the prediction and top SHAP drivers, then writes a short
structured note. It never sees raw sensor data.

**What's not in here**
- No live sensor feed — historical datasets only.
- No real-time alerts — predictions made on demand.
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


# ---------- TABS ----------
tab_turbofan, tab_bearing = st.tabs(["✈  Turbofan (C-MAPSS)", "⚙  Bearing (CWRU)"])


# ===========================================================================
# TAB 1 — TURBOFAN (identical to existing, unchanged)
# ===========================================================================
with tab_turbofan:
    COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    USEFUL = [f"s{i}" for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
    WINDOW = 10
    BASELINE = 63.0

    SENSOR_LABELS = {
        "s2": "LPC outlet total temp",
        "s3": "HPC outlet temp",
        "s4": "LPT coolant temp",
        "s7": "HPC outlet temp",
        "s8": "Physical fan speed",
        "s9": "Physical core speed",
        "s11": "Static core press",
        "s12": "Fan exit temp",
        "s13": "Corrected fan speed",
        "s14": "Corrected core speed",
        "s15": "Bypass duct press",
        "s17": "Engine press ratio",
        "s20": "HPT coolant flow",
        "s21": "Coolant bleed flow",
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

    head_l, head_r = st.columns([3, 2])
    with head_l:
        st.markdown(
            '<div class="pm-header-title"><span class="dot"></span>'
            'Predictive Maintenance AI Platform</div>'
            '<div class="pm-header-sub">Turbofan engine — RUL prediction, explanation, and AI maintenance report</div>',
            unsafe_allow_html=True,
        )
    with head_r:
        c1, c2 = st.columns([1, 1])
        with c1:
            engine_id = st.selectbox("Unit", list(range(1, 101)), index=46, label_visibility="collapsed")
        with c2:
            st.button("Generate Report", use_container_width=True)

    st.markdown("---")

    with st.spinner("Loading model (first run ~20s)..."):
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
    if pred > 80:    bar_color = "#4edea3"; status = "OPTIMAL"
    elif pred > 30:  bar_color = "#ffb95f"; status = "CAUTION"
    else:            bar_color = "#ffb4ab"; status = "CRITICAL"

    left, right = st.columns([5, 7])

    with left:
        st.markdown(f"""
        <div class="pm-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div class="pm-sub" style="margin:0 0 4px 0;">TURBOFAN UNIT ID / C-MAPSS FD001</div>
              <div style="font-size:26px;font-weight:600;letter-spacing:-0.02em;">Engine {engine_id}</div>
              <div class="pm-sub" style="margin-top:4px;">CYCLE #{last_cycle}</div>
            </div>
            <span class="pm-health"><span class="dot"></span>OPERATIONAL</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-card-title">Predicted RUL
            <span class="pm-badge" style="margin-left:auto;">Primary Inferencer</span>
          </div>
          <div style="display:flex;align-items:baseline;">
            <span class="pm-metric" style="color:{bar_color};">{pred:.0f}</span>
            <span class="pm-unit">cycles</span>
            <span class="pm-badge {('green' if pred>80 else 'blue')}" style="margin-left:auto;">
              STATUS: {status} ({'>80' if pred>80 else ('30–80' if pred>30 else '<30')})
            </span>
          </div>
          <div class="pm-bar">
            <div class="pm-bar-fill" style="width:{ratio*100:.1f}%;background:{bar_color};"></div>
          </div>
          <div class="pm-sub" style="display:flex;justify-content:space-between;">
            <span>&lt;30 CRIT</span><span>30–80 CAUTION</span><span>&gt;80 NORMAL</span><span>MAX 125</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        delta = pred - BASELINE
        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-card-title">Population Baseline
            <span class="pm-badge blue" style="margin-left:auto;">{delta:+.0f} cyc vs fleet</span>
          </div>
          <div><span class="pm-metric" style="color:#e5e1e4;">{BASELINE:.0f}</span><span class="pm-unit">cycles</span></div>
          <div class="pm-sub" style="margin-top:6px;">Fleet median across 100 engines</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pm-card" style="background:#2a2a2c;">
          <div class="pm-card-title" style="color:#bec8d2;">True RUL (Answer Key)
            <span class="pm-badge" style="margin-left:auto;">Benchmark</span>
          </div>
          <div><span class="pm-metric muted">{true_rul}</span><span class="pm-unit" style="color:#88929b;">cycles</span></div>
          <div class="pm-sub">Ground truth from test set</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="pm-card-title" style="font-size:20px;font-weight:600;">Why this prediction?</div>', unsafe_allow_html=True)

        names = [f"{p[0].split('_')[0]} ({SENSOR_LABELS.get(p[0].split('_')[0], '')})  [{p[0]}]" for p in pairs][::-1]
        vals  = [p[1] for p in pairs][::-1]
        colors = ["#0EA5E9" if v > 0 else "#ffb4ab" for v in vals]

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1c1b1d")
        ax.set_facecolor("#1c1b1d")
        bars = ax.barh(names, vals, color=colors, height=0.62)
        ax.axvline(0, color="#3e4850", linewidth=1.2, linestyle="--")
        ax.tick_params(colors="#bec8d2", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_xlabel("SHAP value (impact on predicted RUL in cycles)",
                      color="#bec8d2", fontsize=10, labelpad=10)
        for bar, v in zip(bars, vals):
            x = bar.get_width()
            ax.text(x + (0.3 if v > 0 else -0.3), bar.get_y() + bar.get_height()/2,
                    f"{v:+.1f}", va="center",
                    ha="left" if v > 0 else "right",
                    color="#0EA5E9" if v > 0 else "#ffb4ab",
                    fontsize=9, fontfamily="monospace")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown(f'<div class="pm-sub">Base value (population average): {BASELINE:.0f} cycles</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    top8 = tuple((f, round(float(v), 2)) for f, v in pairs[:8])

    with st.spinner("Asking Gemini..."):
        report = cached_report(
            engine_id=int(engine_id),
            predicted_rul=round(pred, 1),
            baseline=BASELINE,
            top_features_tuple=top8,
        )

    if isinstance(report, dict):
        verdict        = report.get("verdict", "UNKNOWN")
        urgency        = report.get("urgency", "Routine")
        summary        = report.get("summary", "—")
        drivers_pos    = report.get("drivers_positive", "—")
        drivers_neg    = report.get("drivers_negative", "—")
        recommendation = report.get("recommendation", "—")
    else:
        verdict, urgency = "UNKNOWN", "Routine"
        summary, drivers_pos, drivers_neg, recommendation = str(report), "—", "—", "—"

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
        AI Maintenance Report
        <span class="pm-badge green" style="margin-left:8px;">Synthesized</span>
        <span class="pm-badge" style="margin-left:auto;background:{vbg};color:{vc};">
          {verdict} · {urgency}
        </span>
      </div>
      <div style="font-size:15px;line-height:1.7;color:#e5e1e4;margin-bottom:18px;">{summary}</div>
      <div class="pm-sr-grid">
        <div><div class="pm-sr-label">Pushed RUL up</div><div class="pm-sr-pos">{drivers_pos}</div></div>
        <div><div class="pm-sr-label">Pushed RUL down</div><div class="pm-sr-neg">{drivers_neg}</div></div>
      </div>
      <div class="pm-sr-action">
        <div class="pm-sr-label" style="margin-bottom:4px;">Recommended action</div>
        <div style="font-size:14px;color:#e5e1e4;">{recommendation}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(f'<div class="pm-provenance">MODEL: XGBoost Regressor · SHAP: TreeExplainer · '
                f'ANALYZED: {now}</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — BEARING (CWRU)
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

    head_l, head_r = st.columns([3, 2])
    with head_l:
        st.markdown(
            '<div class="pm-header-title"><span class="dot"></span>'
            'Predictive Maintenance AI Platform</div>'
            '<div class="pm-header-sub">Bearing — vibration fault classification, explanation, and AI maintenance report</div>',
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
            st.button("Generate Report", key="bearing_gen", use_container_width=True)

    st.markdown("---")

    model_b, feature_names_b, class_names_b = load_bearing_model()
    if model_b is None:
        st.warning(
            "Bearing model not trained yet. Run `python code/train_cwru.py` once, "
            "then reload this page."
        )
        st.stop()

    mat_path = os.path.join(cwru.CWRU_DIR, BEARING_FILES[bearing_label])

    with st.spinner("Analysing vibration signal…"):
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

    # ---- LEFT ----
    with left:
        # Bearing header card
        st.markdown(f"""
        <div class="pm-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div class="pm-sub" style="margin:0 0 4px 0;">BEARING UNIT / CWRU 12k DRIVE-END</div>
              <div style="font-size:26px;font-weight:600;letter-spacing:-0.02em;">
                {bearing_label.split('(')[0].strip()}
              </div>
              <div class="pm-sub" style="margin-top:4px;">{result['n_windows']} WINDOWS ANALYSED</div>
            </div>
            <span class="pm-health"
                  style="background:{'rgba(78,222,163,0.12)' if status_b=='OPTIMAL' else ('rgba(255,185,95,0.12)' if status_b=='CAUTION' else 'rgba(255,180,171,0.12)')};
                         color:{bar_color_b};">
              <span class="dot" style="background:{bar_color_b};"></span>{status_b}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Predicted condition card
        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-card-title">
            <span style="display:inline-flex;align-items:center;gap:6px;">
              <span style="color:{bar_color_b};">◉</span> Predicted condition
            </span>
            <span class="pm-badge" style="margin-left:auto;">Classifier</span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:600;
                      letter-spacing:-0.02em;color:{bar_color_b};margin-bottom:8px;">
            {predicted_name}
          </div>
          <div class="pm-bar">
            <div class="pm-bar-fill" style="width:{confidence*100:.1f}%;background:{bar_color_b};"></div>
          </div>
          <div class="pm-sub">Confidence: {confidence:.1%} &nbsp; (avg across {result['n_windows']} windows)</div>
        </div>
        """, unsafe_allow_html=True)

        # Class probabilities card
        rows_html = ""
        for i, (name, p) in enumerate(zip(class_names_b, mean_proba)):
            border = "" if i == len(class_names_b) - 1 else "border-bottom:1px solid #2a2a2c;"
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:8px 0;{border}">'
                f'<span style="color:#bec8d2;font-size:13px;">{name}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;'
                f'color:#e5e1e4;font-weight:500;">{p:.1%}</span></div>'
            )
        st.markdown(f"""
        <div class="pm-card">
          <div class="pm-card-title">
            <span style="display:inline-flex;align-items:center;gap:6px;">
              <span style="color:#88929b;">▤</span> Class probabilities
            </span>
          </div>
          {rows_html}
        </div>
        """, unsafe_allow_html=True)

    # ---- RIGHT ----
    with right:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">'
            '<div class="pm-card-title" style="font-size:20px;font-weight:600;margin:0;">'
            'Why this prediction?</div>'
            '<span class="pm-badge blue">TreeSHAP Local Attribution</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="display:flex;gap:24px;padding:8px 12px;background:#1c1b1d;'
            'border-radius:6px;margin-bottom:10px;">'
            '<div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#bec8d2;">'
            '<span style="width:12px;height:12px;background:#ffb4ab;border-radius:3px;"></span>'
            'Negative impact <span style="color:#ffb4ab;">(Opposes class)</span></div>'
            '<div style="display:flex;align-items:center;gap:8px;font-size:11px;color:#bec8d2;">'
            '<span style="width:12px;height:12px;background:#0EA5E9;border-radius:3px;"></span>'
            f'Positive impact <span style="color:#0EA5E9;">(Pushes toward {predicted_name})</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        names = [p[0] for p in pairs_b][::-1]
        vals  = [float(p[1]) for p in pairs_b][::-1]
        colors = ["#0EA5E9" if v > 0 else "#ffb4ab" for v in vals]

        fig, ax = plt.subplots(figsize=(8, 5.2), facecolor="#1c1b1d")
        ax.set_facecolor("#1c1b1d")
        bars = ax.barh(names, vals, color=colors, height=0.62)
        ax.axvline(0, color="#88929b", linewidth=1.0, linestyle=(0, (3, 3)))
        ax.tick_params(colors="#bec8d2", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_xlabel(
            f"SHAP value (impact on {predicted_name} probability)",
            color="#bec8d2", fontsize=10, labelpad=10,
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
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown(
            '<div style="text-align:center;font-size:11px;color:#88929b;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-top:6px;">'
            'Vibration features · time-domain statistics + FFT band energies</div>',
            unsafe_allow_html=True,
        )

    # ---- AI REPORT ----
    st.markdown("<br>", unsafe_allow_html=True)
    top8_b = tuple((f, round(float(v), 4)) for f, v in pairs_b[:8])

    with st.spinner("Asking Gemini..."):
        report_b = cached_bearing_report(
            bearing_id=bearing_label,
            predicted_class=predicted_name,
            confidence=round(confidence, 3),
            top_features_tuple=top8_b,
        )

    if isinstance(report_b, dict):
        v_b = report_b.get("verdict", "UNKNOWN")
        u_b = report_b.get("urgency", "Routine")
        s_b = report_b.get("summary", "—")
        dp_b = report_b.get("drivers_positive", "—")
        dn_b = report_b.get("drivers_negative", "—")
        r_b = report_b.get("recommendation", "—")
    else:
        v_b, u_b = "UNKNOWN", "Routine"
        s_b, dp_b, dn_b, r_b = str(report_b), "—", "—", "—"

    vc_b, vbg_b = verdict_colors.get(v_b, verdict_colors["UNKNOWN"])

    st.markdown(f"""
    <div class="pm-card">
      <div class="pm-card-title" style="font-size:18px;font-weight:600;">
        AI Maintenance Report
        <span class="pm-badge green" style="margin-left:8px;">Synthesized</span>
        <span class="pm-badge" style="margin-left:auto;background:{vbg_b};color:{vc_b};">
          {v_b} · {u_b}
        </span>
      </div>
      <div style="font-size:15px;line-height:1.7;color:#e5e1e4;margin-bottom:18px;">{s_b}</div>
      <div class="pm-sr-grid">
        <div>
          <div class="pm-sr-label">Pushed toward {predicted_name}</div>
          <div class="pm-sr-pos">{dp_b}</div>
        </div>
        <div>
          <div class="pm-sr-label">Opposing signals</div>
          <div class="pm-sr-neg">{dn_b}</div>
        </div>
      </div>
      <div class="pm-sr-action">
        <div class="pm-sr-label" style="margin-bottom:4px;">Recommended action</div>
        <div style="font-size:14px;color:#e5e1e4;">{r_b}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f'<div class="pm-provenance">MODEL: XGBoost Classifier · SHAP: TreeExplainer · '
        f'ANALYZED: {now}</div>',
        unsafe_allow_html=True,
    )