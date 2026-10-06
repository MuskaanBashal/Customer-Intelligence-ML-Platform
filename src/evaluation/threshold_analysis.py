from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
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

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    X = df.drop(
        columns=[
            "customerid",
            "churn",
        ]
    )

    y = df["churn"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


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


def build_model(preprocessor):

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


def evaluate_thresholds(
    y_true,
    probabilities,
):

    thresholds = np.arange(
        0.20,
        0.71,
        0.05,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "predicted_churn": int(
                    predictions.sum()
                ),
            }
        )

    return pd.DataFrame(results)


def main():

    print("Loading dataset...")

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(df)

    print(
        f"Training/validation customers: "
        f"{len(X_train):,}"
    )

    print(
        f"Locked test customers: "
        f"{len(X_test):,}"
    )

    print(
        "\nGenerating out-of-fold "
        "probabilities..."
    )

    preprocessor = build_preprocessor(
        X_train
    )

    model = build_model(
        preprocessor
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    print(
        "Out-of-fold probabilities generated."
    )

    results = evaluate_thresholds(
        y_train,
        probabilities,
    )

    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_f1_row = results.loc[
        results["f1"].idxmax()
    ]

    print("\n" + "=" * 80)
    print("BEST THRESHOLD BY F1")
    print("=" * 80)

    print(
        f"Threshold : "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_row['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1_row['f1']:.4f}"
    )

    print(
        f"Customers flagged: "
        f"{int(best_f1_row['predicted_churn'])}"
    )


if __name__ == "__main__":
    main()