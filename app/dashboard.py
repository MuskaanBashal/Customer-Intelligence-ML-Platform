import requests
import streamlit as st
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict",
)


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)


st.title("Customer Churn Intelligence Platform")

st.write(
    "Predict telecommunications customer churn risk "
    "using the deployed machine learning model."
)


# ---------------------------------------------------------
# Customer information
# ---------------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
    )

    seniorcitizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: (
            "Yes" if x == 1 else "No"
        ),
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"],
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"],
    )

    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
    )

    phoneservice = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )


with col2:

    multiplelines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service",
        ],
    )

    internetservice = st.selectbox(
        "Internet Service",
        [
            "Fiber optic",
            "DSL",
            "No",
        ],
    )

    onlinesecurity = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    onlinebackup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    deviceprotection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    techsupport = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )


with col3:

    streamingtv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    streamingmovies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    )

    paperlessbilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    paymentmethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthlycharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0,
    )

    totalcharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0,
        step=10.0,
    )


# ---------------------------------------------------------
# Construct API payload
# ---------------------------------------------------------

customer_data = {

    "gender": gender,
    "seniorcitizen": seniorcitizen,
    "partner": partner,
    "dependents": dependents,
    "tenure": tenure,
    "phoneservice": phoneservice,
    "multiplelines": multiplelines,
    "internetservice": internetservice,
    "onlinesecurity": onlinesecurity,
    "onlinebackup": onlinebackup,
    "deviceprotection": deviceprotection,
    "techsupport": techsupport,
    "streamingtv": streamingtv,
    "streamingmovies": streamingmovies,
    "contract": contract,
    "paperlessbilling": paperlessbilling,
    "paymentmethod": paymentmethod,
    "monthlycharges": monthlycharges,
    "totalcharges": totalcharges,
}


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

st.divider()

predict_button = st.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True,
)


if predict_button:

    try:

        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        probability = result[
            "churn_probability_percent"
        ]

        threshold = (
            result["decision_threshold"]
            * 100
        )

        risk_level = result[
            "risk_level"
        ]

        st.header("Prediction Result")

        metric1, metric2, metric3 = (
            st.columns(3)
        )

        metric1.metric(
            "Churn Probability",
            f"{probability:.2f}%",
        )

        metric2.metric(
            "Decision Threshold",
            f"{threshold:.0f}%",
        )

        metric3.metric(
            "Risk Level",
            risk_level,
        )

        if risk_level == "HIGH":

            st.error(
                "This customer has been classified "
                "as high churn risk."
            )

        else:

            st.success(
                "This customer has been classified "
                "as low churn risk."
            )

        st.progress(
            min(
                probability / 100,
                1.0,
            )
        )
        # -------------------------------------------------
        # Customer-level SHAP explanation
        # -------------------------------------------------

        model_factors = result.get(
            "top_model_factors",
            [],
        )

        if model_factors:

            st.subheader(
                "Why this prediction?"
            )

            st.write(
                "These are the strongest factors "
                "influencing the model's prediction "
                "for this customer."
            )

            friendly_names = {
                "numeric__month_to_month":
                    "Month-to-month contract",
                "numeric__tenure":
                    "Customer tenure",
                "categorical__onlinesecurity_No":
                    "No online security",
                "numeric__fiber_optic":
                    "Fiber optic service",
                "numeric__avg_charge_per_month":
                    "Average charge per month",
                "categorical__techsupport_No":
                    "No technical support",
                "categorical__contract_Month-to-month":
                    "Month-to-month contract",
                "categorical__paymentmethod_Electronic check":
                    "Electronic check payment",
                "numeric__monthlycharges":
                    "Monthly charges",
                "numeric__totalcharges":
                    "Total charges",
            }

            for factor in model_factors:

                raw_name = factor[
                    "feature"
                ]

                display_name = (
                    friendly_names.get(
                        raw_name,
                        raw_name
                        .replace("numeric__", "")
                        .replace("categorical__", "")
                        .replace("_", " ")
                        .title(),
                    )
                )

                shap_value = factor[
                    "shap_value"
                ]

                direction = factor[
                    "direction"
                ]

                if direction == "higher":

                    st.write(
                        f"⬆️ **{display_name}** — "
                        f"pushed predicted churn "
                        f"risk higher "
                        f"(SHAP: {shap_value:+.3f})"
                    )

                else:

                    st.write(
                        f"⬇️ **{display_name}** — "
                        f"pushed predicted churn "
                        f"risk lower "
                        f"(SHAP: {shap_value:+.3f})"
                    )

            st.caption(
                "SHAP values explain how features "
                "influenced this model prediction. "
                "They do not establish that a feature "
                "caused customer churn."
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the prediction API. "
            "Make sure FastAPI is running on "
            "http://127.0.0.1:8000."
        )


    except requests.exceptions.RequestException as error:

        st.error(
            f"Prediction request failed: {error}"
        )


# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------

st.divider()

st.header("Model Performance")

p1, p2, p3, p4 = st.columns(4)

p1.metric(
    "ROC-AUC",
    "84.73%",
)

p2.metric(
    "PR-AUC",
    "66.32%",
)

p3.metric(
    "Recall",
    "72.73%",
)

p4.metric(
    "F1 Score",
    "64.08%",
)


st.caption(
    "Performance metrics are from the locked "
    "held-out test set. Decision threshold = 0.35."
)