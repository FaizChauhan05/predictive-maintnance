import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    min_child_weight=5,
    random_state=42,
)

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
USEFUL = [f"s{i}" for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]

def load_and_label(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    if "train" in path:
        m = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (m - df["cycle"]).clip(upper=125)
    return df

def features_rowwise(df, window=10):
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

            # inside features_rowwise, after computing row["RUL"]:
            if "RUL" in g.columns and row["RUL"] > 125:
                continue   # skip the healthy phase — it's capped anyway
            for c in USEFUL:
                v = tail[c].values
                row[f"{c}_delta10"] = v[-1] - v[0]
                row[f"{c}_delta5"]  = v[-1] - v[len(v)//2]
            out.append(row)
    return pd.DataFrame(out)

train = load_and_label("train_FD001.txt")
test  = load_and_label("test_FD001.txt")

ftr = features_rowwise(train)

low  = ftr[ftr["RUL"] <= 80]
high = ftr[ftr["RUL"] >  80].sample(frac=0.3, random_state=42)
ftr = pd.concat([low, high]).sample(frac=1.0, random_state=42).reset_index(drop=True)
print("Balanced ytr stats:", ftr["RUL"].min(), ftr["RUL"].max(), ftr["RUL"].mean())

fte = features_rowwise(test)

ytr = ftr["RUL"].values
Xtr = ftr.drop(columns=["unit", "cycle", "RUL"]).values

# for test, we still only care about the LAST row per engine
fte_last = fte.sort_values("cycle").groupby("unit").tail(1)
Xte = fte_last.drop(columns=["unit", "cycle"]).values

yte = pd.read_csv("RUL_FD001.txt", sep=r"\s+", header=None).values.ravel()

# sanity check — these MUST match
print("Xtr shape:", Xtr.shape)
print("Xte shape:", Xte.shape)
assert Xtr.shape[1] == Xte.shape[1], "feature count mismatch"

model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(Xtr, ytr)
preds = model.predict(Xte)

print("RMSE:", root_mean_squared_error(yte, preds))
print("MAE :", mean_absolute_error(yte, preds))

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(yte, preds, alpha=0.6)
plt.plot([0, 150], [0, 150], "r--")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title(f"FD001 test set — RMSE={root_mean_squared_error(yte, preds):.1f}")
plt.grid(True)
plt.savefig("preds_vs_true.png", dpi=120)
print("saved preds_vs_true.png")



print("ytr min/max/mean:", ytr.min(), ytr.max(), ytr.mean())
print("ytr first 10:", ytr[:10])
print("ytr last 10 :", ytr[-10:])

# check the feature matrix itself — any all-zero columns?
print("Xtr min/max per column (first 5 cols):")
for j in range(5):
    print(j, Xtr[:, j].min(), Xtr[:, j].max(), Xtr[:, j].std())

# check the test rows specifically
print("Xte min/max per column (first 5 cols):")
for j in range(5):
    print(j, Xte[:, j].min(), Xte[:, j].max(), Xte[:, j].std())