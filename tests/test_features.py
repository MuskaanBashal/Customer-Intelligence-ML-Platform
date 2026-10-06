import pytest
import pandas as pd

from api.main import engineer_features
from api.schemas import CustomerInput


def create_customer(**overrides):

    customer_data = {
        "gender": "Female",
        "seniorcitizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 10,
        "phoneservice": "Yes",
        "multiplelines": "No",
        "internetservice": "Fiber optic",
        "onlinesecurity": "No",
        "onlinebackup": "Yes",
        "deviceprotection": "No",
        "techsupport": "Yes",
        "streamingtv": "Yes",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Credit card (automatic)",
        "monthlycharges": 80.0,
        "totalcharges": 800.0,
    }

    customer_data.update(
        overrides
    )

    return CustomerInput(
        **customer_data
    )


def test_month_to_month_feature():

    customer = create_customer(
        contract="Month-to-month"
    )

    features = engineer_features(
        customer
    )

    assert (
        features["month_to_month"].iloc[0]
        == 1
    )


def test_automatic_payment_feature():

    customer = create_customer(
        paymentmethod=(
            "Credit card (automatic)"
        )
    )

    features = engineer_features(
        customer
    )

    assert (
        features["automatic_payment"].iloc[0]
        == 1
    )


def test_fiber_optic_feature():

    customer = create_customer(
        internetservice="Fiber optic"
    )

    features = engineer_features(
        customer
    )

    assert (
        features["fiber_optic"].iloc[0]
        == 1
    )


def test_internet_feature():

    customer = create_customer(
        internetservice="No",
        onlinesecurity=(
            "No internet service"
        ),
        onlinebackup=(
            "No internet service"
        ),
        deviceprotection=(
            "No internet service"
        ),
        techsupport=(
            "No internet service"
        ),
        streamingtv=(
            "No internet service"
        ),
        streamingmovies=(
            "No internet service"
        ),
    )

    features = engineer_features(
        customer
    )

    assert (
        features["has_internet"].iloc[0]
        == 0
    )


def test_tech_support_feature():

    customer = create_customer(
        techsupport="Yes"
    )

    features = engineer_features(
        customer
    )

    assert (
        features["has_tech_support"].iloc[0]
        == 1
    )


def test_average_charge_per_month():

    customer = create_customer(
        tenure=10,
        totalcharges=800.0,
    )

    features = engineer_features(
        customer
    )

    assert (
        features[
            "avg_charge_per_month"
        ].iloc[0]
        == pytest.approx(80.0)
    )


def test_zero_tenure_average_charge():

    customer = create_customer(
        tenure=0,
        totalcharges=0.0,
    )

    features = engineer_features(
        customer
    )

    assert (
        features[
            "avg_charge_per_month"
        ].iloc[0]
        == 0.0
    )


def test_model_feature_schema():

    customer = create_customer()

    features = engineer_features(
        customer
    )

    assert features.shape[0] == 1

    assert features.shape[1] == 27

def test_api_uses_shared_feature_engineering():
    """API feature engineering should match the shared feature logic."""

    from api.main import engineer_features
    from api.schemas import CustomerInput
    from src.features.build_features import create_features

    customer = CustomerInput(
        gender="Female",
        seniorcitizen=0,
        partner="No",
        dependents="No",
        tenure=2,
        phoneservice="Yes",
        multiplelines="No",
        internetservice="Fiber optic",
        onlinesecurity="No",
        onlinebackup="No",
        deviceprotection="No",
        techsupport="No",
        streamingtv="Yes",
        streamingmovies="Yes",
        contract="Month-to-month",
        paperlessbilling="Yes",
        paymentmethod="Electronic check",
        monthlycharges=89.50,
        totalcharges=179.00,
    )

    api_features = engineer_features(customer)

    raw_df = pd.DataFrame([customer.model_dump()])
    shared_features = create_features(raw_df)

    shared_features = shared_features[
        api_features.columns
    ]

    pd.testing.assert_frame_equal(
        api_features.reset_index(drop=True),
        shared_features.reset_index(drop=True),
    )