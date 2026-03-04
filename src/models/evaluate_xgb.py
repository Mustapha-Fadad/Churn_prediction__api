import sys
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import json

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Fix module path
root_path = Path(__file__).resolve().parents[2]
from src.features.feature_engineering import FeatureEngineer


def evaluate_xgb_model(model_path: str, test_path: str):

    pipeline = joblib.load(model_path)

    df = pd.read_csv(test_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ===============================
    # Probabilities
    # ===============================

    y_prob = pipeline.predict_proba(X)[:, 1]

    print("\nROC-AUC Score:")
    print(roc_auc_score(y, y_prob))

    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title("ROC Curve - Test Set")
    plt.show()

    # ===============================
    # Precision-Recall Curve
    # ===============================

    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
    plt.show()

    print(f"\nPR-AUC Score: {pr_auc:.4f}")

    # ===============================
    # Best Threshold (F1)
    # ===============================

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    print(f"\nBest Threshold (F1 max): {best_threshold:.3f}")
    print(f"Best F1 Score: {best_f1:.3f}")

    y_pred = (y_prob > best_threshold).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # ===============================
    # Save Threshold Metadata
    # ===============================

    metadata = {
        "model": "xgboost",
        "best_threshold": float(best_threshold)
    }

    models_path = root_path / "src" / "artifacts"
    metadata_path = models_path / "xgb_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nThreshold metadata saved at:")
    print(metadata_path)


if __name__ == "__main__":

    base_path = Path(__file__).resolve().parents[2]

    model_path = base_path / "src" / "artifacts" / "xgb_pipeline.pkl"
    test_path = base_path / "src" / "artifacts" / "test_data.csv"

    evaluate_xgb_model(str(model_path), str(test_path))