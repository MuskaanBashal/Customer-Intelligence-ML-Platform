from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "telco_customer_churn.csv"


def profile_dataset() -> None:
    """Run basic data-quality checks on the raw churn dataset."""

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)

    for column in df.columns:
        print(column)

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(df.isna().sum())

    print("\n" + "=" * 60)
    print("UNIQUE VALUES")
    print("=" * 60)

    print(df.nunique())

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    print(df["Churn"].value_counts())
    print("\nPercentages:")
    print((df["Churn"].value_counts(normalize=True) * 100).round(2))

    print("\n" + "=" * 60)
    print("TOTAL CHARGES INSPECTION")
    print("=" * 60)

    numeric_total_charges = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    invalid_total_charges = numeric_total_charges.isna().sum()

    print(f"Values that cannot be converted to numeric: {invalid_total_charges}")


if __name__ == "__main__":
    profile_dataset()