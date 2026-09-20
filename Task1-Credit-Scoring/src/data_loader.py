"""Data loader for German Credit (Statlog) dataset."""
from pathlib import Path
import pandas as pd

COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

GERMAN_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_german_credit(csv_path: str | None = None) -> pd.DataFrame:
    """Load German Credit data.

    Priority:
    1. csv_path if given
    2. data/german.data or data/german_credit.csv if present
    3. Download from UCI archive
    Returns DataFrame with target column 'target' (1=good, 0=bad).
    Original class: 1=good, 2=bad.
    """
    candidates = []
    if csv_path:
        candidates.append(Path(csv_path))
    candidates += [
        DATA_DIR / "german.data",
        DATA_DIR / "german_credit.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".csv":
                df = pd.read_csv(p)
            else:
                df = pd.read_csv(p, sep=r"\s+", header=None, names=COLUMNS)
            return _finalize(df)

    # Download fallback
    print(f"Downloading German Credit from {GERMAN_DATA_URL} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = DATA_DIR / "german.data"
    df = pd.read_csv(GERMAN_DATA_URL, sep=r"\s+", header=None, names=COLUMNS)
    df.to_csv(local, sep=" ", header=False, index=False)
    print(f"Saved to {local}")
    return _finalize(df)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Map class 1->1 (good), 2->0 (bad)
    if "class" in df.columns:
        df["target"] = (df["class"] == 1).astype(int)
        df = df.drop(columns=["class"])
    # Downcast numerics to save RAM (also useful for Home Credit path)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


if __name__ == "__main__":
    df = load_german_credit()
    print(df.shape)
    print(df["target"].value_counts(normalize=True))
