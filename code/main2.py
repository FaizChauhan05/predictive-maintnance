import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
USEFUL = [f"s{i}" for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
WINDOW = 30
BATCH = 128
EPOCHS = 30
LR = 1e-3
DEVICE = "cpu"

# ---------- LOAD ----------
def load(path, labeled):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    if labeled:
        m = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (m - df["cycle"]).clip(upper=125)
    return df

train = load("train_FD001.txt", labeled=True)
test  = load("test_FD001.txt",  labeled=False)
yte   = pd.read_csv("RUL_FD001.txt", sep=r"\s+", header=None).values.ravel()

# ---------- NORMALIZE ----------
# fit scaler on training sensor values only
mu = train[USEFUL].mean().values
sd = train[USEFUL].std().values + 1e-8

def norm(df):
    df = df.copy()
    df[USEFUL] = (df[USEFUL] - mu) / sd
    return df

train = norm(train)
test  = norm(test)

# ---------- SEQUENCES ----------
def make_sequences(df, labeled):
    X, y = [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        vals = g[USEFUL].values
        if labeled:
            rul = g["RUL"].values
            for i in range(WINDOW - 1, len(g)):
                X.append(vals[i - WINDOW + 1 : i + 1])
                y.append(rul[i])
        else:
            # test: only the last window per engine
            if len(g) >= WINDOW:
                X.append(vals[-WINDOW:])
            else:
                # pad short sequences at the start
                pad = np.repeat(vals[:1], WINDOW - len(g), axis=0)
                X.append(np.vstack([pad, vals]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

Xtr, ytr = make_sequences(train, labeled=True)
Xte, _   = make_sequences(test,  labeled=False)

print("Xtr:", Xtr.shape, "Xte:", Xte.shape, "ytr mean:", ytr.mean())

# ---------- MODEL ----------
class RULNet(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, 64)
        last = out[:, -1, :]           # last timestep
        return self.head(last).squeeze(-1)

model = RULNet(Xtr.shape[2]).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

# ---------- TRAIN ----------
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    rmse = (total / len(ds)) ** 0.5
    print(f"epoch {epoch+1:2d}  train RMSE {rmse:.2f}")

# ---------- EVAL ----------
model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(Xte)).numpy()
preds = np.clip(preds, 0, 125)

print("Test RMSE:", root_mean_squared_error(yte, preds))
print("Test MAE :", mean_absolute_error(yte, preds))

plt.figure(figsize=(6,6))
plt.scatter(yte, preds, alpha=0.6)
plt.plot([0,150],[0,150],"r--")
plt.xlabel("True RUL"); plt.ylabel("Predicted RUL")
plt.title(f"LSTM on FD001 — RMSE={root_mean_squared_error(yte, preds):.1f}")
plt.grid(True); plt.savefig("lstm_scatter.png", dpi=120)
print("saved lstm_scatter.png")



import os
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm.pt")
np.save("models/scaler_mu.npy", mu)
np.save("models/scaler_sd.npy", sd)
print("saved models/lstm.pt + scalers")