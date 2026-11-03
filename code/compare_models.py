from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
USEFUL = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
WINDOW_XGB = 10
WINDOW_LSTM = 30


class RULNet(nn.Module):
    def __init__(self, n_features=len(USEFUL)):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_timestep = out[:, -1, :]
        return self.head(last_timestep).squeeze(-1)


def load_datasets():
    train_df = pd.read_csv(DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    max_cycle = train_df.groupby("unit")["cycle"].transform("max")
    train_df["RUL"] = (max_cycle - train_df["cycle"]).clip(upper=125)

    test_df = pd.read_csv(DATA_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    y_true = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None).values.ravel()
    return train_df, test_df, y_true


def get_xgb_model(train_df):
    rows = []
    for unit, g in train_df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        for i in range(len(g)):
            if i < WINDOW_XGB - 1:
                continue
            tail = g.iloc[i - WINDOW_XGB + 1 : i + 1]
            row = {"unit": unit, "cycle": g.loc[i, "cycle"], "RUL": g.loc[i, "RUL"]}
            for c in USEFUL:
                v = tail[c].values
                row[f"{c}_std"] = v.std()
                row[f"{c}_delta"] = v[-1] - v[0]
            rows.append(row)
    ftr = pd.DataFrame(rows)

    low = ftr[ftr["RUL"] <= 80]
    high = ftr[ftr["RUL"] > 80].sample(frac=0.3, random_state=42)
    balanced = pd.concat([low, high]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_train = balanced.drop(columns=["unit", "cycle", "RUL"])
    y_train = balanced["RUL"].values

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def get_lstm_model():
    model = RULNet(n_features=len(USEFUL))
    weights_path = MODELS_DIR / "lstm.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"LSTM weights not found at {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    mu = np.load(MODELS_DIR / "scaler_mu.npy")
    sd = np.load(MODELS_DIR / "scaler_sd.npy")
    return model, mu, sd


def compare_all_engines():
    print("Loading NASA C-MAPSS FD001 test data...")
    train_df, test_df, y_true = load_datasets()

    print("Training / loading XGBoost model...")
    xgb_model = get_xgb_model(train_df)

    print("Loading pretrained LSTM model and scalers...")
    lstm_model, mu, sd = get_lstm_model()

    # 1. XGBoost predictions across all test engines (last 10 cycles per engine)
    xgb_features = []
    engine_ids = []
    for unit, g in test_df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        tail = g.tail(WINDOW_XGB)
        row = {}
        for c in USEFUL:
            v = tail[c].values
            row[f"{c}_std"] = v.std()
            row[f"{c}_delta"] = v[-1] - v[0]
        xgb_features.append(row)
        engine_ids.append(int(unit))

    X_xgb = pd.DataFrame(xgb_features)
    xgb_preds = np.clip(xgb_model.predict(X_xgb), 0, 125)

    # 2. LSTM predictions across all test engines (last 30 normalized cycles per engine)
    test_norm = test_df.copy()
    test_norm[USEFUL] = (test_norm[USEFUL] - mu) / sd

    lstm_windows = []
    for unit, g in test_norm.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        vals = g[USEFUL].values
        if len(g) >= WINDOW_LSTM:
            lstm_windows.append(vals[-WINDOW_LSTM:])
        else:
            pad = np.repeat(vals[:1], WINDOW_LSTM - len(g), axis=0)
            lstm_windows.append(np.vstack([pad, vals]))

    X_lstm = torch.from_numpy(np.array(lstm_windows, dtype=np.float32))
    with torch.no_grad():
        lstm_preds = np.clip(lstm_model(X_lstm).numpy(), 0, 125)

    # 3. Validation checks
    assert len(engine_ids) == 100, f"Expected 100 engines, got {len(engine_ids)}"
    assert len(xgb_preds) == 100, f"Expected 100 XGBoost predictions, got {len(xgb_preds)}"
    assert len(lstm_preds) == 100, f"Expected 100 LSTM predictions, got {len(lstm_preds)}"
    assert len(y_true) == 100, f"Expected 100 true RUL labels, got {len(y_true)}"

    # 4. Error calculations
    lstm_abs_err = np.abs(lstm_preds - y_true)
    xgb_abs_err = np.abs(xgb_preds - y_true)

    better_model = []
    for l_err, x_err in zip(lstm_abs_err, xgb_abs_err):
        if l_err < x_err:
            better_model.append("LSTM")
        elif x_err < l_err:
            better_model.append("XGBoost")
        else:
            better_model.append("Tie")

    df_comp = pd.DataFrame({
        "Engine ID": engine_ids,
        "Actual RUL": y_true.astype(int),
        "LSTM Predicted RUL": np.round(lstm_preds, 1),
        "XGBoost Predicted RUL": np.round(xgb_preds, 1),
        "LSTM Absolute Error": np.round(lstm_abs_err, 1),
        "XGBoost Absolute Error": np.round(xgb_abs_err, 1),
        "Better Model": better_model,
    })

    # 5. Overall metrics
    lstm_rmse = root_mean_squared_error(y_true, lstm_preds)
    lstm_mae = mean_absolute_error(y_true, lstm_preds)
    xgb_rmse = root_mean_squared_error(y_true, xgb_preds)
    xgb_mae = mean_absolute_error(y_true, xgb_preds)

    lstm_wins = (df_comp["Better Model"] == "LSTM").sum()
    xgb_wins = (df_comp["Better Model"] == "XGBoost").sum()
    ties = (df_comp["Better Model"] == "Tie").sum()

    print("\n" + "=" * 50)
    print("NASA C-MAPSS FD001: LSTM vs XGBoost Model Comparison")
    print("=" * 50)
    print(f"Total Test Engines Evaluated: {len(df_comp)}")
    print(f"LSTM    -> RMSE: {lstm_rmse:.2f} cycles | MAE: {lstm_mae:.2f} cycles | Wins: {lstm_wins}/100")
    print(f"XGBoost -> RMSE: {xgb_rmse:.2f} cycles | MAE: {xgb_mae:.2f} cycles | Wins: {xgb_wins}/100")
    print(f"Ties    -> {ties}/100 engines")
    print("=" * 50)

    print("\nFirst 10 Engines Comparison:")
    print(df_comp.head(10).to_string(index=False))

    csv_path = ROOT_DIR / "rul_model_comparison_fd001.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"\nSaved full comparison table to {csv_path}")

    return df_comp, {
        "lstm_rmse": lstm_rmse,
        "lstm_mae": lstm_mae,
        "xgb_rmse": xgb_rmse,
        "xgb_mae": xgb_mae,
    }


if __name__ == "__main__":
    compare_all_engines()

