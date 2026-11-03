from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
USEFUL = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
WINDOW = 30
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cpu"


def load_dataset(filename, is_train=False):
    df = pd.read_csv(DATA_DIR / filename, sep=r"\s+", header=None, names=COLS)
    if is_train:
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (max_cycle - df["cycle"]).clip(upper=125)
    return df


def create_sequences(df, is_train=True):
    X, y = [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        vals = g[USEFUL].values

        if is_train:
            rul_values = g["RUL"].values
            for i in range(WINDOW - 1, len(g)):
                X.append(vals[i - WINDOW + 1 : i + 1])
                y.append(rul_values[i])
        else:
            # Test set: extract only the final operating window per engine
            if len(g) >= WINDOW:
                X.append(vals[-WINDOW:])
            else:
                # Pad short engines at the start by repeating the first row
                pad = np.repeat(vals[:1], WINDOW - len(g), axis=0)
                X.append(np.vstack([pad, vals]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class RULNet(nn.Module):
    def __init__(self, n_features):
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


def main():
    train_df = load_dataset("train_FD001.txt", is_train=True)
    test_df = load_dataset("test_FD001.txt", is_train=False)
    y_test = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None).values.ravel()

    # Z-score normalization fitted only on training set
    mu = train_df[USEFUL].mean().values
    sd = train_df[USEFUL].std().values + 1e-8

    train_norm = train_df.copy()
    test_norm = test_df.copy()
    train_norm[USEFUL] = (train_norm[USEFUL] - mu) / sd
    test_norm[USEFUL] = (test_norm[USEFUL] - mu) / sd

    X_train, y_train = create_sequences(train_norm, is_train=True)
    X_test, _ = create_sequences(test_norm, is_train=False)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train mean: {y_train.mean():.1f}")

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RULNet(n_features=X_train.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        epoch_rmse = (total_loss / len(dataset)) ** 0.5
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{EPOCHS} — Train RMSE: {epoch_rmse:.2f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test)).numpy()
    test_preds = np.clip(test_preds, 0, 125)

    rmse = root_mean_squared_error(y_test, test_preds)
    mae = mean_absolute_error(y_test, test_preds)
    print(f"\nLSTM Test RMSE: {rmse:.2f}")
    print(f"LSTM Test MAE : {mae:.2f}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, test_preds, alpha=0.6, color="#10b981")
    plt.plot([0, 150], [0, 150], "r--", label="Ideal Prediction")
    plt.xlabel("True Remaining Useful Life (Cycles)")
    plt.ylabel("Predicted Remaining Useful Life (Cycles)")
    plt.title(f"LSTM RUL Prediction (FD001)\nRMSE={rmse:.1f}, MAE={mae:.1f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lstm_scatter.png", dpi=120)
    print(f"Saved plot to {PLOTS_DIR / 'lstm_scatter.png'}")

    torch.save(model.state_dict(), MODELS_DIR / "lstm.pt")
    np.save(MODELS_DIR / "scaler_mu.npy", mu)
    np.save(MODELS_DIR / "scaler_sd.npy", sd)
    print(f"Saved model weights and scalers to {MODELS_DIR}")


if __name__ == "__main__":
    main()
