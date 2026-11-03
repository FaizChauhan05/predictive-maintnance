from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
USEFUL = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
WINDOW = 10


def load_dataset(filename, is_train=False):
    df = pd.read_csv(DATA_DIR / filename, sep=r"\s+", header=None, names=COLS)
    if is_train:
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (max_cycle - df["cycle"]).clip(upper=125)
    return df


def extract_features(df, window=WINDOW):
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
                v = tail[c].values
                row[f"{c}_std"] = v.std()
                row[f"{c}_delta"] = v[-1] - v[0]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset("train_FD001.txt", is_train=True)
    test_df = load_dataset("test_FD001.txt", is_train=False)

    train_features = extract_features(train_df)
    low_rul = train_features[train_features["RUL"] <= 80]
    high_rul = train_features[train_features["RUL"] > 80].sample(frac=0.3, random_state=42)
    balanced_train = pd.concat([low_rul, high_rul]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train = balanced_train.drop(columns=["unit", "cycle", "RUL"])
    y_train = balanced_train["RUL"].values

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # 1. Global Feature Importance (Bar Plot)
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_global_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / 'shap_global_bar.png'}")

    # 2. Global Feature Impact & Direction (Beeswarm Plot)
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title("SHAP Feature Summary (Magnitude & Direction)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_global_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / 'shap_global_beeswarm.png'}")

    # 3. Local Explanation for Engine #1
    test_features = extract_features(test_df)
    engine1_features = test_features[test_features["unit"] == 1].tail(1)
    X_engine1 = engine1_features.drop(columns=["unit", "cycle"])

    sv_engine1 = explainer.shap_values(X_engine1)[0]
    pred_rul = model.predict(X_engine1)[0]
    print(f"Engine 1 predicted RUL: {pred_rul:.1f} cycles")

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=sv_engine1,
            base_values=explainer.expected_value,
            data=X_engine1.iloc[0].values,
            feature_names=list(X_engine1.columns),
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_local_engine1.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / 'shap_local_engine1.png'}")


if __name__ == "__main__":
    main()
