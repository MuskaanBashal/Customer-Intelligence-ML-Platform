from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

FIGURE_DIR = BASE_DIR / "docs" / "figures"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("CUSTOMER CHURN EDA")
    print("=" * 60)

    print(f"\nCustomers: {len(df):,}")
    print(f"Overall churn rate: {df['churn'].mean():.2%}")

    print("\nChurn by contract:")
    print(
        df.groupby("contract")["churn"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    print("\nChurn by internet service:")
    print(
        df.groupby("internetservice")["churn"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    print("\nChurn by payment method:")
    print(
        df.groupby("paymentmethod")["churn"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    print("\nAverage numeric features by churn:")
    print(
        df.groupby("churn")[
            ["tenure", "monthlycharges", "totalcharges"]
        ].mean()
    )


def plot_churn_distribution(df: pd.DataFrame) -> None:
    counts = df["churn"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        ["No Churn", "Churn"],
        counts.values
    )

    ax.set_title("Customer Churn Distribution")
    ax.set_ylabel("Number of Customers")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "churn_distribution.png",
        dpi=200
    )

    plt.close(fig)


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    churn_rate = (
        df.groupby("contract")["churn"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        churn_rate.index,
        churn_rate.values
    )

    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("Contract")

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _: f"{value:.0%}"
        )
    )

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "churn_by_contract.png",
        dpi=200
    )

    plt.close(fig)


def plot_tenure_by_churn(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    no_churn = df.loc[
        df["churn"] == 0,
        "tenure"
    ]

    churn = df.loc[
        df["churn"] == 1,
        "tenure"
    ]

    ax.hist(
        [no_churn, churn],
        bins=20,
        label=["No Churn", "Churn"],
        alpha=0.7
    )

    ax.set_title("Tenure Distribution by Churn")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Customers")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "tenure_by_churn.png",
        dpi=200
    )

    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    print_summary(df)

    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_tenure_by_churn(df)

    print(
        f"\nEDA figures saved to:\n{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()