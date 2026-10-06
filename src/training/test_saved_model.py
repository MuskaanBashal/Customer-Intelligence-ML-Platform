from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_xgboost_pipeline.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "model_metadata.joblib"
)


def main():

    print("Loading saved model...")

    model = joblib.load(
        MODEL_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    threshold = metadata[
        "decision_threshold"
    ]

    print("Model loaded successfully.")

    print(
        f"Decision threshold: "
        f"{threshold}"
    )

    # --------------------------------------------------
    # Load example customer
    # --------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    example = df.iloc[[0]].copy()

    customer_id = example[
        "customerid"
    ].iloc[0]

    actual_churn = example[
        "churn"
    ].iloc[0]

    X = example.drop(
        columns=[
            "customerid",
            "churn",
        ]
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    probability = (
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    print("\n" + "=" * 60)
    print("SAVED MODEL INFERENCE TEST")
    print("=" * 60)

    print(
        f"Customer ID: "
        f"{customer_id}"
    )

    print(
        f"Actual churn: "
        f"{actual_churn}"
    )

    print(
        f"Churn probability: "
        f"{probability:.2%}"
    )

    print(
        f"Decision threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Predicted churn: "
        f"{prediction}"
    )

    if prediction == 1:
        risk_label = "HIGH RISK"
    else:
        risk_label = "LOW RISK"

    print(
        f"Risk classification: "
        f"{risk_label}"
    )

    print(
        "\nSaved model inference "
        "completed successfully."
    )


if __name__ == "__main__":
    main()