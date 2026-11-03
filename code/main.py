from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

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
                row[f"{c}_delta10"] = v[-1] - v[0]
                row[f"{c}_delta5"] = v[-1] - v[len(v) // 2]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    train_df = load_dataset("train_FD001.txt", is_train=True)
    test_df = load_dataset("test_FD001.txt", is_train=False)
    y_test = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None).values.ravel()

    train_features = extract_features(train_df)
    test_features = extract_features(test_df)

    # Balance training set: undersample long healthy phase (RUL > 80)
    low_rul = train_features[train_features["RUL"] <= 80]
    high_rul = train_features[train_features["RUL"] > 80].sample(frac=0.3, random_state=42)
    balanced_train = pd.concat([low_rul, high_rul]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train = balanced_train.drop(columns=["unit", "cycle", "RUL"]).values
    y_train = balanced_train["RUL"].values

    # Test set: take the last window of each engine
    test_last_window = test_features.sort_values("cycle").groupby("unit").tail(1)
    X_test = test_last_window.drop(columns=["unit", "cycle"]).values

    print(f"Train samples: {X_train.shape[0]}, Test engines: {X_test.shape[0]}, Features: {X_train.shape[1]}")

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"XGBoost Test RMSE: {rmse:.2f}")
    print(f"XGBoost Test MAE : {mae:.2f}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color="#0284c7")
    plt.plot([0, 150], [0, 150], "r--", label="Ideal Prediction")
    plt.xlabel("True Remaining Useful Life (Cycles)")
    plt.ylabel("Predicted Remaining Useful Life (Cycles)")
    plt.title(f"XGBoost RUL Prediction (FD001)\nRMSE={rmse:.1f}, MAE={mae:.1f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "preds_vs_true.png", dpi=120)
    print(f"Saved plot to {PLOTS_DIR / 'preds_vs_true.png'}")


if __name__ == "__main__":
    main()
