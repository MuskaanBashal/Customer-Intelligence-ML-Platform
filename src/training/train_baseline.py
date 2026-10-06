from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)

MODEL_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Load feature-engineered dataset."""

    return pd.read_csv(DATA_PATH)


def prepare_features(df):
    """Separate identifiers, target and model features."""

    X = df.drop(
        columns=[
            "customerid",
            "churn",
        ]
    )

    y = df["churn"]

    return X, y


def split_data(X, y):
    """Create stratified training and test sets."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor(X):
    """Create preprocessing pipeline."""

    categorical_features = (
        X.select_dtypes(
            include=["object", "string", "category"]
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
                SimpleImputer(strategy="median"),
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

    preprocessor = ColumnTransformer(
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

    return preprocessor


def evaluate_model(
    name,
    model,
    X_test,
    y_test,
):
    """Evaluate a fitted classifier."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    for metric, value in metrics.items():
        if metric != "model":
            print(
                f"{metric:12}: "
                f"{value:.4f}"
            )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
    y_test,
    predictions,
    digits=4,
    zero_division=0,
)
    )

    return metrics


def main():

    print("Loading feature dataset...")

    df = load_data()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Test samples: {len(X_test):,}"
    )

    print(
        f"Training churn rate: "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Test churn rate: "
        f"{y_test.mean():.2%}"
    )

    preprocessor = build_preprocessor(
        X_train
    )

    # -------------------------------------------------
    # Dummy baseline
    # -------------------------------------------------

    dummy_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                DummyClassifier(
                    strategy="prior"
                ),
            ),
        ]
    )

    print("\nTraining dummy baseline...")

    dummy_model.fit(
        X_train,
        y_train,
    )

    dummy_metrics = evaluate_model(
        "Dummy Baseline",
        dummy_model,
        X_test,
        y_test,
    )

    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_metrics = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test,
    )

    # -------------------------------------------------
    # Model comparison
    # -------------------------------------------------

    comparison = pd.DataFrame(
        [
            dummy_metrics,
            logistic_metrics,
        ]
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "logistic_regression.joblib"
    )

    joblib.dump(
        logistic_model,
        model_path,
    )

    print(
        f"\nModel saved to:\n{model_path}"
    )


if __name__ == "__main__":
    main()