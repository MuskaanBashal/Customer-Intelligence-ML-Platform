from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)

REPORT_PATH = (
    BASE_DIR
    / "docs"
    / "drift_report.csv"
)


NUMERICAL_FEATURES = [
    "tenure",
    "monthlycharges",
    "totalcharges",
    "num_services",
    "avg_charge_per_month",
]


def calculate_psi(
    reference,
    current,
    bins=10,
):

    reference = np.asarray(reference)
    current = np.asarray(current)

    # Reference-based quantile boundaries
    boundaries = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1),
        )
    )

    if len(boundaries) < 2:
        return 0.0

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=boundaries,
    )

    current_counts, _ = np.histogram(
        current,
        bins=boundaries,
    )

    reference_percent = (
        reference_counts
        / len(reference)
    )

    current_percent = (
        current_counts
        / len(current)
    )

    # Avoid division/log of zero
    epsilon = 0.0001

    reference_percent = np.maximum(
        reference_percent,
        epsilon,
    )

    current_percent = np.maximum(
        current_percent,
        epsilon,
    )

    psi = np.sum(
        (
            current_percent
            - reference_percent
        )
        * np.log(
            current_percent
            / reference_percent
        )
    )

    return float(psi)


def classify_drift(psi):

    if psi < 0.10:
        return "LOW"

    if psi < 0.25:
        return "MODERATE"

    return "HIGH"


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    # --------------------------------------------------
    # Simulate historical vs production populations
    # --------------------------------------------------

    reference = df.sample(
        frac=0.70,
        random_state=42,
    )

    current = df.drop(
        reference.index
    ).copy()

    print(
        f"Reference customers: "
        f"{len(reference):,}"
    )

    print(
        f"Current customers: "
        f"{len(current):,}"
    )

    print(
        "\nSimulating production drift..."
    )

    # Artificial drift is added intentionally so that
    # the monitoring system can be demonstrated.

    current["monthlycharges"] = (
        current["monthlycharges"]
        * 1.20
    )

    current["tenure"] = (
        current["tenure"]
        * 0.70
    )

    # --------------------------------------------------
    # Calculate PSI
    # --------------------------------------------------

    results = []

    for feature in NUMERICAL_FEATURES:

        psi = calculate_psi(
            reference[feature],
            current[feature],
        )

        results.append(
            {
                "feature": feature,
                "psi": psi,
                "drift_level": (
                    classify_drift(psi)
                ),
            }
        )

    report = pd.DataFrame(
        results
    ).sort_values(
        "psi",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("DATA DRIFT REPORT")
    print("=" * 60)

    print(
        report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    print(
        f"\nDrift report saved to:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
