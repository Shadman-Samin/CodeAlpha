"""Evaluate classifier: Precision, Recall, F1, ROC-AUC + plots."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, classification_report,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def evaluate(y_true, y_pred, y_proba, model_name: str) -> dict:
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred, target_names=["bad(0)", "good(1)"], zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["bad", "good"], yticklabels=["bad", "good"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"confusion_matrix_{model_name}.png", dpi=120)
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(f"ROC Curve - {model_name} (AUC={metrics['roc_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"roc_{model_name}.png", dpi=120)
    plt.close()

    with open(ASSETS_DIR / f"metrics_{model_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
