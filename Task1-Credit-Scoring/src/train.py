"""Train credit scoring models: LogReg, DecisionTree, RandomForest.

Usage:
    python src/train.py
"""
import sys
from pathlib import Path
import json
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_german_credit
from features import build_preprocessor, get_feature_lists
from evaluate import evaluate, ASSETS_DIR

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    df = load_german_credit()
    print(f"Dataset: {df.shape}, good-rate={df['target'].mean():.3f}")
    X = df.drop(columns=["target"])
    y = df["target"]
    numeric, categorical = get_feature_lists(df)
    print(f"Numeric: {numeric}\nCategorical: {len(categorical)} cols")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": Pipeline([
            ("prep", build_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "DecisionTree": GridSearchCV(
            Pipeline([
                ("prep", build_preprocessor(numeric, categorical)),
                ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
            ]),
            param_grid={"clf__max_depth": [4, 6, 8, None],
                        "clf__min_samples_leaf": [1, 5]},
            cv=5, scoring="roc_auc", n_jobs=-1,
        ),
        "RandomForest": GridSearchCV(
            Pipeline([
                ("prep", build_preprocessor(numeric, categorical)),
                ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
            ]),
            param_grid={"clf__n_estimators": [200, 300],
                        "clf__max_depth": [8, None]},
            cv=5, scoring="roc_auc", n_jobs=-1,
        ),
    }

    all_metrics = {}
    best_auc, best_name, best_model = 0, "", None
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        clf = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        if hasattr(model, "best_params_"):
            print(f"Best params: {model.best_params_}")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, y_pred, y_proba, name)
        all_metrics[name] = m
        if m["roc_auc"] > best_auc:
            best_auc, best_name, best_model = m["roc_auc"], name, clf

    joblib.dump(best_model, ASSETS_DIR / "best_model.pkl")
    with open(ASSETS_DIR / "metrics_all.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nBest: {best_name} (ROC-AUC={best_auc:.4f}) -> assets/best_model.pkl")


if __name__ == "__main__":
    main()
