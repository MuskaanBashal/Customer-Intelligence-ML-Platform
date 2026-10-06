from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the feature-engineered customer dataset."""

    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Prepare features and target
# ---------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    """
    Remove identifiers and separate features from the
    churn target.
    """

    X = df.drop(
        columns=[
            "customerid",
            "churn",
        ]
    )

    y = df["churn"]

    return X, y


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

def create_train_test_split(X, y):
    """
    Reserve the final test set.

    Model comparison will only use the training portion.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def build_preprocessor(X_train):
    """Build numeric and categorical preprocessing."""

    categorical_features = (
        X_train.select_dtypes(
            include=["object", "string", "category"]
        )
        .columns
        .tolist()
    )

    numerical_features = (
        X_train.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    print(
        f"Numerical features: "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
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


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

def build_models():
    """Define candidate classification models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------

def compare_models(
    X_train,
    y_train,
    preprocessor,
):
    """
    Evaluate candidate models using stratified
    5-fold cross-validation.
    """

    models = build_models()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = []

    for model_name, classifier in models.items():

        print("\n" + "=" * 60)
        print(f"Evaluating {model_name}")
        print("=" * 60)

        pipeline = Pipeline(
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

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        model_result = {
            "model": model_name,
            "accuracy_mean": scores[
                "test_accuracy"
            ].mean(),
            "precision_mean": scores[
                "test_precision"
            ].mean(),
            "recall_mean": scores[
                "test_recall"
            ].mean(),
            "f1_mean": scores[
                "test_f1"
            ].mean(),
            "roc_auc_mean": scores[
                "test_roc_auc"
            ].mean(),
            "pr_auc_mean": scores[
                "test_pr_auc"
            ].mean(),
            "roc_auc_std": scores[
                "test_roc_auc"
            ].std(),
        }

        results.append(model_result)

        print(
            f"Accuracy : "
            f"{model_result['accuracy_mean']:.4f}"
        )

        print(
            f"Precision: "
            f"{model_result['precision_mean']:.4f}"
        )

        print(
            f"Recall   : "
            f"{model_result['recall_mean']:.4f}"
        )

        print(
            f"F1       : "
            f"{model_result['f1_mean']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{model_result['roc_auc_mean']:.4f}"
        )

        print(
            f"PR-AUC   : "
            f"{model_result['pr_auc_mean']:.4f}"
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading feature dataset...")

    df = load_data()

    X, y = prepare_features(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_train_test_split(X, y)

    print(
        f"Training/validation samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Locked test samples: "
        f"{len(X_test):,}"
    )

    print(
        f"Training churn rate: "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Test churn rate: "
        f"{y_test.mean():.2%}"
    )

    print(
        "\nThe test set will NOT be used "
        "for model comparison."
    )

    preprocessor = build_preprocessor(
        X_train
    )

    results = compare_models(
        X_train,
        y_train,
        preprocessor,
    )

    results = results.sort_values(
        by="roc_auc_mean",
        ascending=False,
    )

    print("\n" + "=" * 90)
    print("CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 90)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()