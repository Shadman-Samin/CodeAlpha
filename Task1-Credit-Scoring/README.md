# Credit Scoring Model — CodeAlpha ML Task 1

Predict an individual's creditworthiness (good/bad) using past financial data.
Student ID: `CA/DF1/298776`

## Dataset
**German Credit (Statlog)** — 1000 rows, 20 features + target.
- Good: 700 (70%), Bad: 300 (30%)
- Features: duration, credit_amount, age, checking_status, credit_history, purpose, savings_status, employment, etc.
- Source: UCI / OpenML `credit-g` (saved locally as `data/german_credit.csv`, git-ignored)
- Target: `target` (1=good, 0=bad)

Why German Credit: fast, no auth, fits CodeAlpha key features (income, debts, payment history). Pipeline downcasts dtypes so the same code scales to Home Credit `application_train.csv`.

## Approach
1. Stratified 80/20 split
2. ColumnTransformer: StandardScaler (7 numerics) + OneHot (13 categoricals)
3. class_weight=balanced for imbalance
4. Models (all required by task doc):
   - Logistic Regression (baseline)
   - Decision Tree + GridSearch (max_depth, min_samples_leaf, 5-fold ROC-AUC)
   - Random Forest + GridSearch (n_estimators, max_depth, 5-fold ROC-AUC)

## Results (test set, 200 rows)

| Model | Accuracy | Precision* | Recall* | F1* | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.675 | 0.838 | 0.664 | 0.741 | **0.760** |
| DecisionTree | 0.595 | 0.883 | 0.486 | 0.627 | 0.711 |
| RandomForest | **0.715** | 0.812 | **0.771** | **0.791** | **0.781** |

\*Precision/Recall/F1 for good-credit class; see `assets/` confusion matrices + per-class reports in console output.
Best model saved: `assets/best_model.pkl` (RandomForest, 300 trees, max_depth=8).

Plots: `assets/confusion_matrix_*.png`, `assets/roc_*.png`
Metrics JSON: `assets/metrics_all.json`

## How to run
```powershell
pip install -r requirements.txt
python src/train.py
```
Outputs to `assets/`. No large data download needed — `data/german_credit.csv` already present locally.

## Feature engineering notes
- One-hot for checking_status, credit_history, purpose, savings_status, employment, personal_status, etc.
- Scaling only numerics: duration, credit_amount, age, etc.
- No leakage: preprocessor fit inside Pipeline, GridSearchCV on train only.

## Repo for submission
Push this folder content to GitHub as `CodeAlpha_CreditScoring` (per CodeAlpha rule `CodeAlpha_ProjectName`), then LinkedIn video + submission form via WhatsApp group.
