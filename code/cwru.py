import os
from pathlib import Path
import numpy as np
from scipy.fft import rfft, rfftfreq
import joblib

ROOT = Path(__file__).resolve().parent.parent
CWRU_DIR = str(ROOT / "data" / "cwru")
MODELS_DIR = str(ROOT / "models")

WINDOW = 1024
OVERLAP = 0.5
SAMPLE_RATE = 12000

CWRU_FILES = {
    "Normal":     ["1797_Normal.npz"],
    "Inner Race": ["1797_IR_7_DE12.npz"],
    "Ball":       ["1797_B_7_DE12.npz"],
    "Outer Race": ["1797_OR@6_7_DE12.npz"],
}
CLASS_NAMES = list(CWRU_FILES.keys())


def load_signal(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Signal file not found: {path}")
    data = np.load(path)
    return data["DE"].flatten().astype(np.float32)


def windows_from_signal(signal, window=WINDOW, overlap=OVERLAP):
    step = int(window * (1 - overlap))
    for start in range(0, len(signal) - window + 1, step):
        yield signal[start : start + window]


def window_features(signal, fs=SAMPLE_RATE):
    eps = 1e-12
    feats = {}

    # 1. Time-Domain Statistical Features
    mean = signal.mean()
    std = signal.std() + eps
    rms = np.sqrt((signal ** 2).mean()) + eps
    peak = np.abs(signal).max()
    abs_mean = np.abs(signal).mean() + eps

    feats["mean"] = mean
    feats["std"] = std
    feats["rms"] = rms
    feats["peak"] = peak
    feats["ptp"] = signal.max() - signal.min()
    feats["kurtosis"] = ((signal - mean) ** 4).mean() / (std ** 4)
    feats["skew"] = ((signal - mean) ** 3).mean() / (std ** 3)
    feats["crest"] = peak / rms
    feats["shape"] = rms / abs_mean
    feats["impulse"] = peak / abs_mean

    # 2. Frequency-Domain FFT Features
    n_samples = len(signal)
    fft_spectrum = np.abs(rfft(signal)) / n_samples
    freqs = rfftfreq(n_samples, d=1 / fs)
    total_power = (fft_spectrum ** 2).sum() + eps

    bands = [(0, 500), (500, 1500), (1500, 3000), (3000, 4500), (4500, 6000)]
    for i, (low, high) in enumerate(bands):
        mask = (freqs >= low) & (freqs < high)
        feats[f"band{i}_energy"] = float((fft_spectrum[mask] ** 2).sum() / total_power)

    fft_sum = fft_spectrum.sum() + eps
    feats["spec_centroid"] = float((freqs * fft_spectrum).sum() / fft_sum)
    feats["spec_spread"] = float(np.sqrt(
        (((freqs - feats["spec_centroid"]) ** 2) * fft_spectrum).sum() / fft_sum
    ))

    return feats


def build_dataset(cwru_dir=CWRU_DIR):
    X_rows, y = [], []
    feature_names = None

    for class_idx, (label, files) in enumerate(CWRU_FILES.items()):
        for fname in files:
            path = os.path.join(cwru_dir, fname)
            if not os.path.exists(path):
                continue
            signal = load_signal(path)
            for window in windows_from_signal(signal):
                feats = window_features(window)
                if feature_names is None:
                    feature_names = list(feats.keys())
                X_rows.append([feats[k] for k in feature_names])
                y.append(class_idx)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, feature_names, CLASS_NAMES


def train_classifier(X, y):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(set(y)),
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    return model


def save_artifacts(model, feature_names, class_names, out_dir=MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names, "class_names": class_names},
        os.path.join(out_dir, "cwru_xgb.pkl"),
    )
    print(f"Saved bearing artifacts to {out_dir}/cwru_xgb.pkl")


def load_artifacts(out_dir=MODELS_DIR):
    return joblib.load(os.path.join(out_dir, "cwru_xgb.pkl"))


def predict_file(model, feature_names, path):
    signal = load_signal(path)
    X_file = []
    for window in windows_from_signal(signal):
        feats = window_features(window)
        X_file.append([feats[k] for k in feature_names])
    X_file = np.array(X_file, dtype=np.float32)

    proba = model.predict_proba(X_file)
    mean_proba = proba.mean(axis=0)
    majority = int(np.argmax(mean_proba))

    # Representative window: window whose probability distribution is closest to the average
    dist = np.abs(proba - mean_proba).sum(axis=1)
    rep_idx = int(np.argmin(dist))

    return {
        "n_windows": int(len(X_file)),
        "majority_class": majority,
        "confidence": float(mean_proba[majority]),
        "mean_proba": mean_proba.tolist(),
        "representative_X": X_file[rep_idx : rep_idx + 1],
    }
