import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cwru import build_dataset, train_classifier, save_artifacts


def main():
    print("Extracting features from CWRU bearing dataset...")
    X, y, feature_names, class_names = build_dataset()
    print(f"Dataset shape: {X.shape[0]} windows, {X.shape[1]} features")
    for i, name in enumerate(class_names):
        print(f"  Class {i} ({name}): {(y == i).sum()} windows")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining XGBoost multi-class classifier...")
    model = train_classifier(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, predictions, target_names=class_names))

    save_artifacts(model, feature_names, class_names)


if __name__ == "__main__":
    main()
