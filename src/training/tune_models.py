from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
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
        columns=["customerid", "churn"]
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


def tune_logistic(
    X_train,
    y_train,
    preprocessor,
    cv,
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


def tune_xgboost(
    X_train,
    y_train,
    preprocessor,
    cv,
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [
            200,
            400,
        ],
        "classifier__max_depth": [
            3,
            4,
        ],
        "classifier__learning_rate": [
            0.03,
            0.05,
        ],
        "classifier__subsample": [
            0.8,
            1.0,
        ],
        "classifier__colsample_bytree": [
            0.8,
            1.0,
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


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
        f"Training/validation samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Locked test samples: "
        f"{len(X_test):,}"
    )

    print(
        "\nThe locked test set will NOT "
        "be used during tuning."
    )

    preprocessor = build_preprocessor(
        X_train
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------
    # Logistic Regression
    # -----------------------------------------------

    print("\n" + "=" * 60)
    print("TUNING LOGISTIC REGRESSION")
    print("=" * 60)

    logistic_search = tune_logistic(
        X_train,
        y_train,
        preprocessor,
        cv,
    )

    print(
        "\nBest Logistic parameters:"
    )

    print(
        logistic_search.best_params_
    )

    print(
        f"Best Logistic CV ROC-AUC: "
        f"{logistic_search.best_score_:.4f}"
    )

    # -----------------------------------------------
    # XGBoost
    # -----------------------------------------------

    print("\n" + "=" * 60)
    print("TUNING XGBOOST")
    print("=" * 60)

    xgb_search = tune_xgboost(
        X_train,
        y_train,
        preprocessor,
        cv,
    )

    print(
        "\nBest XGBoost parameters:"
    )

    print(
        xgb_search.best_params_
    )

    print(
        f"Best XGBoost CV ROC-AUC: "
        f"{xgb_search.best_score_:.4f}"
    )


if __name__ == "__main__":
    main()