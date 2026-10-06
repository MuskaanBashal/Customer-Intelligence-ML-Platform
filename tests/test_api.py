from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_CUSTOMER = {
    "gender": "Female",
    "seniorcitizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 2,
    "phoneservice": "Yes",
    "multiplelines": "No",
    "internetservice": "Fiber optic",
    "onlinesecurity": "No",
    "onlinebackup": "No",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "Yes",
    "streamingmovies": "Yes",
    "contract": "Month-to-month",
    "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check",
    "monthlycharges": 89.50,
    "totalcharges": 179.00,
}


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_prediction_endpoint():

    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "decision_threshold" in data
    assert "predicted_churn" in data
    assert "top_model_factors" in data

    assert (
        0
        <= data["churn_probability"]
        <= 1
    )

    assert data[
        "predicted_churn"
    ] in [0, 1]

    assert len(
        data["top_model_factors"]
    ) > 0


def test_invalid_negative_tenure():

    invalid_customer = (
        VALID_CUSTOMER.copy()
    )

    invalid_customer[
        "tenure"
    ] = -10

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422


def test_invalid_contract():

    invalid_customer = (
        VALID_CUSTOMER.copy()
    )

    invalid_customer[
        "contract"
    ] = "Random Contract"

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422