from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI

from api.schemas import CustomerInput
from src.features.build_features import create_features


BASE_DIR = Path(__file__).resolve().parents[1]

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


# ---------------------------------------------------------
# Load production artifacts once at startup/import
# ---------------------------------------------------------

model = joblib.load(
    MODEL_PATH
)

metadata = joblib.load(
    METADATA_PATH
)

DECISION_THRESHOLD = metadata[
    "decision_threshold"
]
preprocessor = model.named_steps[
    "preprocessor"
]

classifier = model.named_steps[
    "classifier"
]

feature_names = (
    preprocessor.get_feature_names_out()
)

explainer = shap.TreeExplainer(
    classifier
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production API for predicting "
        "telecommunications customer churn risk."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def engineer_features(
    customer: CustomerInput,
) -> pd.DataFrame:
    """Convert API input into the model's engineered feature set."""

    data = customer.model_dump()

    df = pd.DataFrame([data])

    # Use the same feature-engineering logic as training.
    df = create_features(df)

    # Match the exact feature schema/order used by the trained model.
    df = df[metadata["input_features"]]

    return df
def explain_prediction(
    features: pd.DataFrame,
    top_n: int = 5,
):

    transformed = (
        preprocessor.transform(
            features
        )
    )

    shap_values = explainer(
        transformed
    )

    values = shap_values.values[0]

    contributions = []

    for feature_name, value in zip(
        feature_names,
        values,
    ):

        contributions.append(
            {
                "feature": feature_name,
                "shap_value": float(value),
                "absolute_shap": float(
                    abs(value)
                ),
            }
        )

    contributions.sort(
        key=lambda item: item[
            "absolute_shap"
        ],
        reverse=True,
    )

    top_contributions = (
        contributions[:top_n]
    )

    for item in top_contributions:

        item.pop(
            "absolute_shap"
        )

        item["shap_value"] = round(
            item["shap_value"],
            4,
        )

        item["direction"] = (
            "higher"
            if item["shap_value"] > 0
            else "lower"
        )

    return top_contributions


# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": (
            "Customer Churn Prediction API"
        ),
        "status": "running",
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": metadata["model_name"],
    }


@app.post("/predict")
def predict_churn(
    customer: CustomerInput,
):

    features = engineer_features(
        customer
    )
    explanation = explain_prediction(
        features
    )

    probability = float(
        model.predict_proba(
            features
        )[0, 1]
    )

    predicted_churn = int(
        probability
        >= DECISION_THRESHOLD
    )

    if predicted_churn == 1:
        risk_level = "HIGH"
    else:
        risk_level = "LOW"

    return {
        "churn_probability": round(
            probability,
            4,
        ),
        "churn_probability_percent": round(
            probability * 100,
            2,
        ),
        "decision_threshold": (
            DECISION_THRESHOLD
        ),
        "predicted_churn": (
            predicted_churn
        ),
        "risk_level": risk_level,
        "top_model_factors": explanation,
    }