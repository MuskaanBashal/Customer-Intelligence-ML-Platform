from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

FIGURE_DIR = BASE_DIR / "docs" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20


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


def build_model():

    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
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
        f"Explanation customers: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------
    # Fit preprocessing using training data only
    # --------------------------------------------------

    print("\nFitting preprocessing pipeline...")

    preprocessor = build_preprocessor(
        X_train
    )

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print(
        f"Features after encoding: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------
    # Train tuned XGBoost
    # --------------------------------------------------

    print(
        "\nTraining tuned XGBoost model..."
    )

    model = build_model()

    model.fit(
        X_train_transformed,
        y_train,
    )

    print("Model training complete.")

    # --------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_test_transformed
    )

    shap_values.feature_names = (
        list(feature_names)
    )

    print("SHAP values calculated.")

    # --------------------------------------------------
    # Global feature importance
    # --------------------------------------------------

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    global_path = (
        FIGURE_DIR
        / "shap_global_importance.png"
    )

    plt.savefig(
        global_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nGlobal SHAP plot saved to:\n"
        f"{global_path}"
    )

    # --------------------------------------------------
    # SHAP beeswarm
    # --------------------------------------------------

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    beeswarm_path = (
        FIGURE_DIR
        / "shap_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSHAP beeswarm saved to:\n"
        f"{beeswarm_path}"
    )

    # --------------------------------------------------
    # Explain highest-risk test customer
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_test_transformed
    )[:, 1]

    highest_risk_position = (
        probabilities.argmax()
    )

    customer_id = df.loc[
        X_test.index[
            highest_risk_position
        ],
        "customerid",
    ]

    churn_probability = probabilities[
        highest_risk_position
    ]

    print("\n" + "=" * 60)
    print("HIGHEST-RISK TEST CUSTOMER")
    print("=" * 60)

    print(
        f"Customer ID: "
        f"{customer_id}"
    )

    print(
        f"Predicted churn probability: "
        f"{churn_probability:.2%}"
    )

    # --------------------------------------------------
    # Local waterfall explanation
    # --------------------------------------------------

    shap.plots.waterfall(
        shap_values[
            highest_risk_position
        ],
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    local_path = (
        FIGURE_DIR
        / "shap_high_risk_customer.png"
    )

    plt.savefig(
        local_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nLocal SHAP explanation saved to:\n"
        f"{local_path}"
    )

    print("\nExplainability complete.")


if __name__ == "__main__":
    main()