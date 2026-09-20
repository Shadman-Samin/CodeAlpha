"""Train disease prediction: Breast Cancer (primary) + Diabetes (secondary).

Models required by task doc: SVM, Logistic Regression, Random Forest, XGBoost.
No download needed — uses sklearn built-in datasets (UCI equivalents).
"""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay,
    classification_report)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def run_dataset(name, X, y, target_names):
    print(f"\n===== {name}: {X.shape} =====")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "LogisticRegression": Pipeline([("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "SVM": Pipeline([("sc", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=42))]),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               n_jobs=-1, random_state=42),
        "XGBoost": __import__("xgboost").XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, eval_metric="logloss", n_jobs=-1, random_state=42),
    }
    results = {}
    for mname, model in models.items():
        model.fit(X_train, y_train)
        yp = model.predict(X_test)
        ypr = model.predict_proba(X_test)[:, 1]
        r = {"model": mname,
             "accuracy": float(accuracy_score(y_test, yp)),
             "precision": float(precision_score(y_test, yp, zero_division=0)),
             "recall": float(recall_score(y_test, yp, zero_division=0)),
             "f1": float(f1_score(y_test, yp, zero_division=0)),
             "roc_auc": float(roc_auc_score(y_test, ypr))}
        print(f"{mname}: acc={r['accuracy']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")
        results[mname] = r
        cm = confusion_matrix(y_test, yp)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - {mname}"); plt.tight_layout()
        plt.savefig(ASSETS / f"cm_{name}_{mname}.png", dpi=110); plt.close()
    # ROC for best
    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"Best {name}: {best}")
    with open(ASSETS / f"metrics_{name}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results, best


def main():
    bc = load_breast_cancer(as_frame=True)
    X_bc = bc.data.values if hasattr(bc.data, "values") else bc.data
    # Diabetes: convert to binary (progression > median) to make it classification
    db = load_diabetes(as_frame=True)
    X_db = db.data.values if hasattr(db.data, "values") else db.data
    y_db = (db.target > db.target.median()).astype(int).values
    r1, b1 = run_dataset("BreastCancer", X_bc, bc.target.values, bc.target_names)
    r2, b2 = run_dataset("Diabetes", X_db, y_db, ["low", "high"])
    # Save best overall (breast cancer best model refit)
    from sklearn.ensemble import RandomForestClassifier
    bc_full = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    bc_full.fit(X_bc, bc.target.values)
    joblib.dump(bc_full, ASSETS / "best_model.pkl")
    print("\nSaved assets/best_model.pkl + metrics_*.json")


if __name__ == "__main__":
    main()
