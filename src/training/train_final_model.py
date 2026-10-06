from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = (
    MODEL_DIR
    / "churn_xgboost_pipeline.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "model_metadata.joblib"
)

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.35


def load_data():
    """Load the feature-engineered dataset."""

    return pd.read_csv(DATA_PATH)


def prepare_training_data(df):
    """
    Prepare all available labelled data for final
    production-model training.
    """

    X = df.drop(
        columns=[
            "customerid",
            "churn",
        ]
    )

    y = df["churn"]

    return X, y


def build_preprocessor(X):

    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numerical_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


def build_pipeline(X):

    preprocessor = build_preprocessor(X)

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def main():

    print("Loading training data...")

    df = load_data()

    X, y = prepare_training_data(df)

    print(
        f"Training customers: {len(X):,}"
    )

    print(
        f"Input features: {X.shape[1]}"
    )

    print(
        f"Churn rate: {y.mean():.2%}"
    )

    print(
        "\nBuilding production pipeline..."
    )

    model = build_pipeline(X)

    print(
        "Training final production model..."
    )

    model.fit(
        X,
        y,
    )

    print("Training complete.")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # --------------------------------------------------
    # Save model metadata
    # --------------------------------------------------

    metadata = {
        "model_name": "XGBoost Customer Churn Classifier",
        "decision_threshold": DECISION_THRESHOLD,
        "input_features": list(X.columns),
        "target": "churn",
        "training_rows": len(X),
        "random_state": RANDOM_STATE,
        "model_parameters": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    }

    joblib.dump(
        metadata,
        METADATA_PATH,
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{METADATA_PATH}"
    )

    print(
        f"\nProduction decision threshold: "
        f"{DECISION_THRESHOLD}"
    )

    print(
        "\nProduction model packaging complete."
    )


if __name__ == "__main__":
    main()
