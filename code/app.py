import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from xgboost import XGBRegressor
from dotenv import load_dotenv

import cwru
from llm_report import generate_report, generate_classification_report

load_dotenv()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(
    page_title="Predictive Maintenance AI Platform",
    page_icon="🛠️",
    layout="wide"
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1a1a24;
        border: 1px solid #2d2d3d;
        padding: 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

TURBOFAN_SENSOR_NAMES = {
    "s2": "LPC Outlet Temp",
    "s3": "HPC Outlet Temp",
    "s4": "LPT Coolant Temp",
    "s7": "HPC Outlet Temp (Static)",
    "s8": "Physical Fan Speed",
    "s9": "Physical Core Speed",
    "s11": "Static Core Pressure",
    "s12": "Fan Exit Temp",
    "s13": "Corrected Fan Speed",
    "s14": "Corrected Core Speed",
    "s15": "Bypass Duct Pressure",
    "s17": "Engine Pressure Ratio",
    "s20": "HPT Coolant Flow",
    "s21": "Coolant Bleed Flow",
}

BEARING_FEATURE_NAMES = {
    "mean": "Signal Mean (DC Offset)",
    "std": "Signal Variability (Std Dev)",
    "rms": "Signal Energy / Intensity (RMS)",
    "peak": "Max Vibration Peak",
    "ptp": "Peak-to-Peak Amplitude",
    "kurtosis": "Spike Sharpness (Kurtosis)",
    "skew": "Signal Asymmetry (Skewness)",
    "crest": "Crest Factor (Peak/RMS)",
    "shape": "Shape Factor",
    "impulse": "Impulse Factor",
    "band0_energy": "Low Frequency Energy (0-500 Hz)",
    "band1_energy": "Mid-Low Freq Energy (500-1500 Hz)",
    "band2_energy": "Mid Freq Energy (1500-3000 Hz)",
    "band3_energy": "Mid-High Freq Energy (3000-4500 Hz)",
    "band4_energy": "High Freq Energy (4500-6000 Hz)",
    "spec_centroid": "Spectral Center of Mass",
    "spec_spread": "Spectral Spread",
}


def format_turbofan_feature(raw_feature_name):
    parts = raw_feature_name.split("_", 1)
    sensor_code = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    sensor_name = TURBOFAN_SENSOR_NAMES.get(sensor_code, sensor_code)
    stat = "Change" if suffix == "delta" else "Variability" if suffix == "std" else suffix
    return f"{sensor_name} ({stat})"


def format_bearing_feature(raw_feature_name):
    return BEARING_FEATURE_NAMES.get(raw_feature_name, raw_feature_name)


def plot_horizontal_shap(features, values, xlabel, title):
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#141419")
    ax.set_facecolor("#141419")

    y_pos = list(range(len(features)))[::-1]
    plot_values = values[::-1]
    plot_labels = features[::-1]

    colors = ["#38bdf8" if v >= 0 else "#f87171" for v in plot_values]
    bars = ax.barh(y_pos, plot_values, color=colors, height=0.55)

    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_labels, color="#e2e8f0", fontsize=9.5)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=9)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=9.5, labelpad=8)
    ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold", pad=12)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#27273a", linestyle=":", linewidth=0.8)

    for bar, val in zip(bars, plot_values):
        offset = 0.15 if val >= 0 else -0.15
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}",
            va="center",
            ha=ha,
            color="#e2e8f0",
            fontsize=8.5,
            fontweight="bold"
        )

    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("Project Overview")
    st.markdown("""
    **Predictive Maintenance AI Platform** combines machine learning and explainability:
    
    1. **Turbofan Engine Health**
       - Predicts **Remaining Useful Life (RUL)** in operating cycles.
       - Uses an **XGBoost Regressor** trained on the NASA C-MAPSS dataset.
       - Explains predictions using **TreeSHAP** feature attribution.
    
    2. **Bearing Fault Classification**
       - Diagnoses rotating bearing condition from raw vibration signals.
       - Extracts 15 statistical and frequency features per window.
       - Uses an **XGBoost Classifier** trained on the CWRU bearing dataset.
    
    3. **AI Maintenance Interpretation**
       - Synthesizes model outputs and SHAP features into an advisory report using Gemini.
    """)
    st.divider()
    st.caption("Predictive Maintenance AI Platform")

st.title("🛠️ Predictive Maintenance AI Platform")
st.markdown("Monitor equipment condition, understand key drivers, and review AI maintenance recommendations.")

tab_turbofan, tab_bearing = st.tabs(["✈️ Turbofan Engine Module", "⚙️ Bearing Vibration Module"])

# --- Turbofan Engine Module ---
with tab_turbofan:
    COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    USEFUL = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    WINDOW = 10
    BASELINE = 63.0

    @st.cache_data
    def load_data():
        train = pd.read_csv(DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COLS)
        max_cycle = train.groupby("unit")["cycle"].transform("max")
        train["RUL"] = (max_cycle - train["cycle"]).clip(upper=125)
        test = pd.read_csv(DATA_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COLS)
        rul_true = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
        return train, test, rul_true

    def features_rowwise(df, window=WINDOW):
        rows = []
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
                    vals = tail[c].values
                    row[f"{c}_std"] = vals.std()
                    row[f"{c}_delta"] = vals[-1] - vals[0]
                rows.append(row)
        return pd.DataFrame(rows)

    @st.cache_resource
    def train_model():
        train, _, _ = load_data()
        ftr = features_rowwise(train)
        low = ftr[ftr["RUL"] <= 80]
        high = ftr[ftr["RUL"] > 80].sample(frac=0.3, random_state=42)
        balanced = pd.concat([low, high]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        X = balanced.drop(columns=["unit", "cycle", "RUL"])
        y = balanced["RUL"].values
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
            vals = tail[c].values
            row[f"{c}_std"] = vals.std()
            row[f"{c}_delta"] = vals[-1] - vals[0]
        return pd.DataFrame([row]), int(g["cycle"].iloc[-1])

    @st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
    def cached_report(engine_id, predicted_rul, baseline, top_features_tuple):
        return generate_report(
            engine_id=engine_id,
            predicted_rul=predicted_rul,
            baseline=baseline,
            top_features=list(top_features_tuple),
        )

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        engine_id = st.selectbox(
            "Select Turbofan Engine from Test Fleet:",
            options=list(range(1, 101)),
            index=46,
            help="Choose an engine ID from the NASA C-MAPSS FD001 test dataset."
        )
    with col_btn:
        st.write("")
        generate_btn = st.button("Generate AI Maintenance Report", key="turbofan_report_btn", use_container_width=True)

    with st.spinner("Loading engine model and datasets..."):
        model, explainer = train_model()
        _, test_df, rul_true_df = load_data()

    x_engine, last_cycle = features_for_engine(test_df, engine_id)
    if x_engine is None:
        st.error("Not enough cycle history to extract window features for this engine.")
        st.stop()

    pred_rul = float(model.predict(x_engine)[0])
    shap_vals = explainer.shap_values(x_engine)[0]
    true_rul = int(rul_true_df.iloc[engine_id - 1]["RUL"])
    pred_error = pred_rul - true_rul

    if pred_rul > 80:
        health_status = "Optimal"
        status_color = "normal"
    elif pred_rul > 30:
        health_status = "Caution"
        status_color = "off"
    else:
        health_status = "Critical"
        status_color = "inverse"

    st.subheader(f"Engine #{engine_id} — Health & Life Prediction")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="Estimated Remaining Life",
            value=f"{pred_rul:.0f} cycles",
            delta=f"Status: {health_status}",
            delta_color=status_color,
            help="Predicted number of operating cycles remaining until service is required."
        )
    with m2:
        diff_fleet = pred_rul - BASELINE
        st.metric(
            label="Fleet Median Remaining Life",
            value=f"{BASELINE:.0f} cycles",
            delta=f"{diff_fleet:+.0f} cycles vs. fleet median",
            delta_color="normal" if diff_fleet >= 0 else "inverse",
            help="Median remaining life across the reference fleet (63 cycles)."
        )
    with m3:
        st.metric(
            label="Actual Benchmark (Test Dataset)",
            value=f"{true_rul} cycles",
            delta=f"Model Error: {pred_error:+.0f} cycles ({'over' if pred_error > 0 else 'under'}estimated)",
            delta_color="off",
            help="Ground truth benchmark from test dataset. Shown for model evaluation only; hidden from the model during prediction."
        )

    health_ratio = max(0.0, min(1.0, pred_rul / 125.0))
    st.progress(health_ratio)
    st.caption(
        f"**Operating Cycle #{last_cycle}** | Remaining life scale: 0 to 125 cycles maximum. "
        "One cycle represents a full flight take-off, cruise, and landing sequence."
    )

    st.divider()

    col_chart, col_details = st.columns([3, 2])

    all_pairs = sorted(zip(x_engine.columns, shap_vals), key=lambda item: abs(item[1]), reverse=True)
    top5_pairs = all_pairs[:5]
    top5_names = [format_turbofan_feature(p[0]) for p in top5_pairs]
    top5_vals = [float(p[1]) for p in top5_pairs]

    with col_chart:
        st.subheader("What influenced the estimated remaining life?")
        fig_rul = plot_horizontal_shap(
            features=top5_names,
            values=top5_vals,
            xlabel="SHAP Impact (cycles added or subtracted)",
            title="Top 5 Sensor Contributions to RUL"
        )
        st.pyplot(fig_rul, use_container_width=True)
        st.caption("🟦 **Blue bars** pushed the estimate higher (healthier). 🟥 **Red bars** lowered the estimate (wear indicated).")

    with col_details:
        st.subheader("Understanding this Prediction")
        st.markdown(f"""
        - **Engine Current State:** Completed **{last_cycle}** operational cycles.
        - **Estimated Remaining Life:** The XGBoost regression model estimates **{pred_rul:.0f} operating cycles** remaining.
        - **Key Sensor Reading:** **{top5_names[0]}** had the single largest impact on the estimate ({top5_vals[0]:+.1f} cycles).
        - **Evaluation vs Benchmark:** The actual remaining life is **{true_rul} cycles**, resulting in an error of **{pred_error:+.1f} cycles**.
        """)

        with st.expander("How does TreeSHAP work here?"):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** calculates the exact contribution of each sensor feature to the prediction:
            - The model starts from a baseline average prediction (**63 cycles**).
            - Each sensor reading either increases or decreases that estimate based on trained decision trees.
            - Features are statistical indicators, not standalone causes of mechanical failure.
            """)

    st.divider()
    st.subheader("AI Maintenance Interpretation")

    top8_tuple = tuple((f, round(float(v), 2)) for f, v in all_pairs[:8])
    with st.spinner("Generating AI maintenance interpretation..."):
        report = cached_report(
            engine_id=int(engine_id),
            predicted_rul=round(pred_rul, 1),
            baseline=BASELINE,
            top_features_tuple=top8_tuple,
        )

    if isinstance(report, dict):
        summary = report.get("summary", "No summary available.")
        pos_drivers = report.get("drivers_positive", "None identified.")
        neg_drivers = report.get("drivers_negative", "None identified.")
        recommendation = report.get("recommendation", "Continue routine monitoring.")
        verdict = report.get("verdict", "CAUTION")
        urgency = report.get("urgency", "Routine")
    else:
        summary, pos_drivers, neg_drivers, recommendation = str(report), "—", "—", "—"
        verdict, urgency = "UNKNOWN", "Routine"

    with st.container():
        st.markdown(f"**Status Verdict:** `{verdict}` | **Urgency:** `{urgency}`")
        st.write(summary)

        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            st.markdown("**Positive Contributing Factors:**")
            st.write(pos_drivers)
        with rep_c2:
            st.markdown("**Negative Contributing Factors:**")
            st.write(neg_drivers)

        st.info(f"💡 **Recommended Next Step:** {recommendation}")
        st.caption("⚠️ Advisory notice: This interpretation is synthesized by Gemini from model outputs and is intended for decision support, not automatic control.")

# --- Bearing Vibration Module ---
with tab_bearing:
    BEARING_FILES = {
        "Normal Baseline (Healthy)": "1797_Normal.npz",
        "Inner Race Fault": "1797_IR_7_DE12.npz",
        "Ball Fault": "1797_B_7_DE12.npz",
        "Outer Race Fault": "1797_OR@6_7_DE12.npz",
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

    col_b_sel, col_b_btn = st.columns([3, 1])
    with col_b_sel:
        selected_signal_label = st.selectbox(
            "Select Bearing Vibration Signal:",
            options=list(BEARING_FILES.keys()),
            index=1,
            help="Select a vibration acceleration signal from the CWRU Bearing Data Center."
        )
    with col_b_btn:
        st.write("")
        generate_b_btn = st.button("Generate AI Maintenance Report", key="bearing_report_btn", use_container_width=True)

    model_b, feature_names_b, class_names_b = load_bearing_model()
    if model_b is None:
        st.warning("Bearing model not found. Please ensure `models/cwru_xgb.pkl` exists.")
        st.stop()

    file_path = os.path.join(cwru.CWRU_DIR, BEARING_FILES[selected_signal_label])
    with st.spinner("Analyzing vibration signal windows..."):
        result_b = cwru.predict_file(model_b, feature_names_b, file_path)

    majority_class_idx = result_b["majority_class"]
    predicted_class_name = class_names_b[majority_class_idx]
    confidence_score = result_b["confidence"]
    mean_probabilities = result_b["mean_proba"]
    rep_X = result_b["representative_X"]

    explainer_b = shap.TreeExplainer(model_b)
    shap_out_b = explainer_b.shap_values(rep_X)
    if isinstance(shap_out_b, list):
        sv_window = shap_out_b[majority_class_idx][0]
    elif shap_out_b.ndim == 3:
        sv_window = shap_out_b[0, :, majority_class_idx]
    else:
        sv_window = shap_out_b[0]

    all_bearing_pairs = sorted(zip(feature_names_b, sv_window), key=lambda item: abs(item[1]), reverse=True)
    top5_bearing_pairs = all_bearing_pairs[:5]
    top5_b_names = [format_bearing_feature(p[0]) for p in top5_bearing_pairs]
    top5_b_vals = [float(p[1]) for p in top5_bearing_pairs]

    st.subheader(f"Vibration Analysis — {selected_signal_label}")

    mb1, mb2, mb3 = st.columns(3)
    with mb1:
        st.metric(
            label="Detected Bearing Condition",
            value=predicted_class_name,
            help="Condition classified by the multi-class XGBoost model."
        )
    with mb2:
        st.metric(
            label="Model Confidence",
            value=f"{confidence_score:.1%}",
            help="Average probability for the detected condition across all signal windows."
        )
    with mb3:
        st.metric(
            label="Windows Analyzed",
            value=f"{result_b['n_windows']} segments",
            help="Total 1024-sample vibration segments analyzed with 50% overlap."
        )

    if predicted_class_name == "Normal":
        st.success("✅ **Healthy Condition:** Vibration patterns are within expected baseline limits.")
    else:
        st.warning(f"⚠️ **Fault Detected ({predicted_class_name}):** Irregular vibration patterns identified.")

    with st.expander("What do these bearing fault categories mean?"):
        st.markdown("""
        - **Normal Baseline:** Smooth rotation with no defects; vibration is low and evenly distributed across frequencies.
        - **Inner Race Fault:** Defect on the inner raceway ring; generates periodic shock pulses when balls pass over it.
        - **Ball Fault:** Flaw or spall on a rolling element ball; causes rotational impact pulses.
        - **Outer Race Fault:** Damage on the stationary outer bearing ring; produces distinct frequency spikes.
        """)

    st.divider()

    col_b_chart1, col_b_chart2 = st.columns(2)

    with col_b_chart1:
        st.subheader("Model Scores for Each Condition")
        fig_prob, ax_prob = plt.subplots(figsize=(6, 2.8), facecolor="#141419")
        ax_prob.set_facecolor("#141419")

        y_indices = list(range(len(class_names_b)))[::-1]
        probs_pct = [p * 100 for p in mean_probabilities][::-1]
        labels_rev = class_names_b[::-1]
        bar_colors = ["#38bdf8" if name == predicted_class_name else "#475569" for name in labels_rev]

        p_bars = ax_prob.barh(y_indices, probs_pct, color=bar_colors, height=0.5)
        ax_prob.set_yticks(y_indices)
        ax_prob.set_yticklabels(labels_rev, color="#e2e8f0", fontsize=9.5)
        ax_prob.set_xlim(0, 105)
        ax_prob.set_xlabel("Probability Score (%)", color="#94a3b8", fontsize=9)
        ax_prob.set_title("Class Probability Distribution", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)

        for spine in ax_prob.spines.values():
            spine.set_visible(False)
        ax_prob.tick_params(axis="x", colors="#94a3b8", labelsize=8.5)

        for bar, val in zip(p_bars, probs_pct):
            ax_prob.text(
                val + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                color="#e2e8f0",
                fontsize=8.5,
                fontweight="bold"
            )

        fig_prob.tight_layout()
        st.pyplot(fig_prob, use_container_width=True)
        st.caption("Highlighted bar shows the highest-scoring detected condition (sums to 100%).")

    with col_b_chart2:
        st.subheader("Which vibration patterns influenced this?")
        fig_b_shap = plot_horizontal_shap(
            features=top5_b_names,
            values=top5_b_vals,
            xlabel=f"Contribution to {predicted_class_name} Score",
            title=f"Top 5 Vibration Feature Influences"
        )
        st.pyplot(fig_b_shap, use_container_width=True)
        st.caption(f"🟦 **Blue** supported the **{predicted_class_name}** class; 🟥 **Red** argued against it.")

    with st.expander("Understanding extracted vibration features"):
        st.markdown("""
        - **Signal Energy / Intensity (RMS):** Root Mean Square measures the overall vibration power. Higher RMS indicates stronger mechanical vibration.
        - **Spike Sharpness (Kurtosis):** Quantifies how peaky or impulsive the signal is. Values above 3.0 strongly indicate defect impact shocks.
        - **Frequency Band Energy:** Divides the FFT spectrum into 5 bands. Faults typically concentrate energy in specific frequency ranges.
        - **Spectral Center of Mass:** The energy-weighted average frequency; shifts higher when high-frequency impact harmonics appear.
        """)

    st.divider()
    st.subheader("AI Maintenance Interpretation")

    top8_b_tuple = tuple((f, round(float(v), 4)) for f, v in all_bearing_pairs[:8])
    with st.spinner("Generating AI maintenance interpretation..."):
        report_b = cached_bearing_report(
            bearing_id=selected_signal_label,
            predicted_class=predicted_class_name,
            confidence=round(confidence_score, 3),
            top_features_tuple=top8_b_tuple,
        )

    if isinstance(report_b, dict):
        summary_b = report_b.get("summary", "No summary available.")
        pos_b = report_b.get("drivers_positive", "None identified.")
        neg_b = report_b.get("drivers_negative", "None identified.")
        rec_b = report_b.get("recommendation", "Continue routine monitoring.")
        verdict_b = report_b.get("verdict", "CAUTION")
        urgency_b = report_b.get("urgency", "Routine")
    else:
        summary_b, pos_b, neg_b, rec_b = str(report_b), "—", "—", "—"
        verdict_b, urgency_b = "UNKNOWN", "Routine"

    with st.container():
        st.markdown(f"**Status Verdict:** `{verdict_b}` | **Urgency:** `{urgency_b}`")
        st.write(summary_b)

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.markdown(f"**Patterns supporting {predicted_class_name}:**")
            st.write(pos_b)
        with b_c2:
            st.markdown("**Opposing / atypical patterns:**")
            st.write(neg_b)

        st.info(f"💡 **Recommended Next Step:** {rec_b}")
        st.caption("⚠️ Advisory notice: This interpretation is synthesized by Gemini from model outputs and is intended for decision support, not automatic control.")

st.divider()
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.caption(f"Predictive Maintenance AI Platform | Analyzed at {now_str} | Models: XGBoost Regressor & Classifier | XAI: TreeSHAP")
