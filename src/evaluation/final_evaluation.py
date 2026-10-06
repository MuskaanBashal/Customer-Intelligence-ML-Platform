from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Selected using out-of-fold training predictions
DECISION_THRESHOLD = 0.35


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
        f"Training customers: "
        f"{len(X_train):,}"
    )

    print(
        f"Final test customers: "
        f"{len(X_test):,}"
    )

    print(
        "\nTraining final tuned XGBoost model..."
    )

    preprocessor = build_preprocessor(
        X_train
    )

    model = build_model(
        preprocessor
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ---------------------------------------------
    # Generate probabilities
    # ---------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------
    # Apply selected threshold
    # ---------------------------------------------

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    # ---------------------------------------------
    # Evaluation metrics
    # ---------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("FINAL LOCKED TEST RESULTS")
    print("=" * 70)

    print(
        f"Decision threshold : "
        f"{DECISION_THRESHOLD:.2f}"
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision          : "
        f"{precision:.4f}"
    )

    print(
        f"Recall             : "
        f"{recall:.4f}"
    )

    print(
        f"F1                 : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC             : "
        f"{pr_auc:.4f}"
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

    # ---------------------------------------------
    # Prediction output
    # ---------------------------------------------

    prediction_output = pd.DataFrame(
        {
            "customerid": df.loc[
                X_test.index,
                "customerid",
            ],
            "actual_churn": y_test,
            "churn_probability": probabilities,
            "predicted_churn": predictions,
        }
    )

    prediction_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "final_test_predictions.csv"
    )

    prediction_output.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"\nPredictions saved to:\n"
        f"{prediction_path}"
    )


if __name__ == "__main__":
    main()