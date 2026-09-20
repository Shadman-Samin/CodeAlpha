"""Preprocessing: one-hot categoricals + scale numerics."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]


def get_feature_lists(df: pd.DataFrame):
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in df.columns if c not in numeric + ["target"]]
    return numeric, categorical


def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
