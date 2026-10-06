from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    BASE_DIR / "data" / "raw" / "telco_customer_churn.csv"
)

PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "telco_customer_churn_clean.csv"
)


def load_data() -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""

    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the raw dataset."""

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Standardise column names
    # ---------------------------------------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # ---------------------------------------------------------
    # 2. Convert TotalCharges to numeric
    # ---------------------------------------------------------
    df["totalcharges"] = pd.to_numeric(
        df["totalcharges"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 3. Handle new customers with tenure = 0
    # ---------------------------------------------------------
    new_customer_mask = (
        (df["tenure"] == 0)
        & (df["totalcharges"].isna())
    )

    df.loc[new_customer_mask, "totalcharges"] = 0.0

    # ---------------------------------------------------------
    # 4. Validate remaining missing TotalCharges
    # ---------------------------------------------------------
    remaining_missing = df["totalcharges"].isna().sum()

    if remaining_missing > 0:
        raise ValueError(
            f"Unexpected missing TotalCharges values: "
            f"{remaining_missing}"
        )

    # ---------------------------------------------------------
    # 5. Convert target to binary
    # ---------------------------------------------------------
    churn_mapping = {
        "No": 0,
        "Yes": 1,
    }

    df["churn"] = df["churn"].map(churn_mapping)

    if df["churn"].isna().any():
        raise ValueError(
            "Unexpected values found in churn target."
        )

    df["churn"] = df["churn"].astype(int)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run basic validation checks after preprocessing."""

    if df.empty:
        raise ValueError("Processed dataset is empty.")

    if df["customerid"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if df["customerid"].isna().any():
        raise ValueError("Missing customer IDs detected.")

    if df.isna().any().any():
        missing = df.isna().sum()
        missing = missing[missing > 0]

        raise ValueError(
            f"Missing values remain:\n{missing}"
        )

    if not set(df["churn"].unique()).issubset({0, 1}):
        raise ValueError("Churn target must contain only 0 and 1.")

    if (df["tenure"] < 0).any():
        raise ValueError("Negative tenure detected.")

    if (df["monthlycharges"] < 0).any():
        raise ValueError("Negative monthly charges detected.")

    if (df["totalcharges"] < 0).any():
        raise ValueError("Negative total charges detected.")


def save_data(df: pd.DataFrame) -> None:
    """Save the cleaned dataset."""

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )


def main() -> None:
    print("Starting preprocessing...")

    df = load_data()

    print(f"Raw dataset shape: {df.shape}")

    df = clean_data(df)

    validate_data(df)

    save_data(df)

    print(f"Processed dataset shape: {df.shape}")
    print(
        f"Missing values: "
        f"{df.isna().sum().sum()}"
    )
    print(
        f"Duplicate customer IDs: "
        f"{df['customerid'].duplicated().sum()}"
    )
    print(
        f"Churn distribution:\n"
        f"{df['churn'].value_counts()}"
    )
    print(
        f"Processed data saved to:\n"
        f"{PROCESSED_DATA_PATH}"
    )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()