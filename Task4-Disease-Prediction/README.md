# Disease Prediction from Medical Data — CodeAlpha ML Task 4

Student ID: `CA/DF1/298776`

## Objective
Predict disease possibility from structured medical data (symptoms, age, test results).

## Datasets (UCI equivalents, no download)
- **Primary: Breast Cancer Wisconsin** (569 x 30) — malignant/benign
- **Secondary: Diabetes progression binarized** (442 x 10)
- Uses sklearn built-in loads = same UCI sources cited in task doc. Compatible with Heart Disease / Pima CSV drop-in.

## Approach
Standardized pipeline + 4 required algorithms: Logistic Regression, SVM (prob=True), Random Forest (300 trees), XGBoost (300 estimators). Stratified 80/20 split.

## Results (verified on this PC)

BreastCancer:
- LogisticRegression: acc 0.956, f1 0.965, **AUC 0.995** (best)
- SVM: acc 0.965, f1 0.972, AUC 0.994
- RandomForest: acc 0.947, AUC 0.994
- XGBoost: acc 0.956, AUC 0.995

Diabetes (binarized):
- RandomForest: acc 0.753, f1 0.766, AUC 0.823
- LogisticRegression: acc 0.742, AUC 0.826

Artifacts: `assets/cm_*.png`, `assets/metrics_*.json`, `assets/best_model.pkl`

## Run
```powershell
pip install -r requirements.txt
python src/train.py
```

## Submit
GitHub repo: `CodeAlpha_DiseasePrediction`
