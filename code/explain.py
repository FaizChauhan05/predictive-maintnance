import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ---------- CONFIG ----------
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
USEFUL = [f"s{i}" for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
WINDOW = 10


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
            for c in USEFUL:
                v = tail[c].values
                row[f"{c}_std"]   = v.std()
                row[f"{c}_delta"] = v[-1] - v[0]
            out.append(row)
    return pd.DataFrame(out)


# ---------- LOAD + LABEL TRAIN ----------
train = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=COLS)
m = train.groupby("unit")["cycle"].transform("max")
train["RUL"] = (m - train["cycle"]).clip(upper=125)

# ---------- FEATURES ----------
ftr = features_rowwise(train)

# balance the RUL distribution (same as the XGBoost run)
low  = ftr[ftr["RUL"] <= 80]
high = ftr[ftr["RUL"] >  80].sample(frac=0.3, random_state=42)
ftr  = pd.concat([low, high]).sample(frac=1.0, random_state=42).reset_index(drop=True)

Xtr = ftr.drop(columns=["unit", "cycle", "RUL"])
ytr = ftr["RUL"].values

# ---------- TRAIN XGBOOST ----------
model = XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0,
    random_state=42,
)
model.fit(Xtr, ytr)

# ---------- SHAP ----------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xtr)

# 1) Global bar — which features matter most on average
plt.figure()
shap.summary_plot(shap_values, Xtr, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_global_bar.png", dpi=120, bbox_inches="tight")
print("saved shap_global_bar.png")

# 2) Global beeswarm — direction of each feature's effect
plt.figure()
shap.summary_plot(shap_values, Xtr, show=False)
plt.tight_layout()
plt.savefig("shap_global_beeswarm.png", dpi=120, bbox_inches="tight")
print("saved shap_global_beeswarm.png")

# ---------- LOCAL: one test engine ----------
test = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None, names=COLS)
fte = features_rowwise(test).sort_values("cycle").groupby("unit").tail(1)
Xte = fte.drop(columns=["unit", "cycle"])

one = Xte.iloc[[0]]
sv_one = explainer.shap_values(one)[0]
pred = model.predict(one)[0]
top = sorted(zip(one.columns, sv_one), key=lambda x: abs(x[1]), reverse=True)[:8]

print(f"\nEngine {int(fte.iloc[0]['unit'])} — predicted RUL: {pred:.1f}")
for f, v in top:
    print(f"  {f:20s}  SHAP {v:+.2f}")

# 3) Waterfall for that engine
plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=sv_one,
        base_values=explainer.expected_value,
        data=one.iloc[0].values,
        feature_names=list(one.columns),
    ),
    show=False,
)
plt.tight_layout()
plt.savefig("shap_local_engine1.png", dpi=120, bbox_inches="tight")
print("saved shap_local_engine1.png")