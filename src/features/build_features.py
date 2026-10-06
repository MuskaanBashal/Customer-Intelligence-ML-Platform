from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)


SERVICE_COLUMNS = [
    "phoneservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
]


def load_data() -> pd.DataFrame:
    """Load the cleaned customer dataset."""
    return pd.read_csv(INPUT_PATH)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level features for churn modelling."""

    df = df.copy()

    # --------------------------------------------------
    # 1. Customer tenure groups
    # --------------------------------------------------
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=[
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months",
        ],
    )

    # --------------------------------------------------
    # 2. Number of active services
    # --------------------------------------------------
    df["num_services"] = sum(
        df[column].eq("Yes").astype(int)
        for column in SERVICE_COLUMNS
    )

    # --------------------------------------------------
    # 3. Automatic payment indicator
    # --------------------------------------------------
    automatic_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    df["automatic_payment"] = (
        df["paymentmethod"]
        .isin(automatic_methods)
        .astype(int)
    )

    # --------------------------------------------------
    # 4. Month-to-month contract indicator
    # --------------------------------------------------
    df["month_to_month"] = (
        df["contract"]
        .eq("Month-to-month")
        .astype(int)
    )

    # --------------------------------------------------
    # 5. Fiber optic customer indicator
    # --------------------------------------------------
    df["fiber_optic"] = (
        df["internetservice"]
        .eq("Fiber optic")
        .astype(int)
    )

    # --------------------------------------------------
    # 6. Customer has internet
    # --------------------------------------------------
    df["has_internet"] = (
        df["internetservice"]
        .ne("No")
        .astype(int)
    )

    # --------------------------------------------------
    # 7. Customer has technical support
    # --------------------------------------------------
    df["has_tech_support"] = (
        df["techsupport"]
        .eq("Yes")
        .astype(int)
    )

    # --------------------------------------------------
    # 8. Average historical charge per tenure month
    # --------------------------------------------------
    df["avg_charge_per_month"] = (
        df["totalcharges"]
        .div(df["tenure"].replace(0, pd.NA))
        .fillna(0)
    )

    return df


def validate_features(df: pd.DataFrame) -> None:
    """Validate engineered features."""

    required_features = [
        "tenure_group",
        "num_services",
        "automatic_payment",
        "month_to_month",
        "fiber_optic",
        "has_internet",
        "has_tech_support",
        "avg_charge_per_month",
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing engineered features: {missing_features}"
        )

    if df[required_features].isna().any().any():
        raise ValueError(
            "Missing values detected in engineered features."
        )


def save_features(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


def main() -> None:
    print("Starting feature engineering...")

    df = load_data()

    print(f"Input shape: {df.shape}")

    df = create_features(df)

    validate_features(df)

    save_features(df)

    print(f"Output shape: {df.shape}")

    print("\nEngineered features:")
    print(
        df[
            [
                "tenure_group",
                "num_services",
                "automatic_payment",
                "month_to_month",
                "fiber_optic",
                "has_internet",
                "has_tech_support",
                "avg_charge_per_month",
            ]
        ].head()
    )

    print(
        "\nFeature dataset saved to:"
        f"\n{OUTPUT_PATH}"
    )

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()