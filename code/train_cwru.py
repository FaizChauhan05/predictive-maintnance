"""
train_cwru.py — one-time training script. Run once, produces models/cwru_xgb.pkl.
"""
from cwru import build_dataset, train_classifier, save_artifacts
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y, feature_names, class_names = build_dataset()
print(f"X: {X.shape}  y: {y.shape}")
for i, n in enumerate(class_names):
    print(f"  {n}: {(y == i).sum()} samples")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = train_classifier(X_tr, y_tr)
preds = model.predict(X_te)

acc = accuracy_score(y_te, preds)
print(f"\nTest accuracy: {acc:.4f}\n")
print(classification_report(y_te, preds, target_names=class_names))

save_artifacts(model, feature_names, class_names)