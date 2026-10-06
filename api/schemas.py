from typing import Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):

    gender: Literal[
        "Male",
        "Female",
    ]

    seniorcitizen: Literal[0, 1]

    partner: Literal[
        "Yes",
        "No",
    ]

    dependents: Literal[
        "Yes",
        "No",
    ]

    tenure: int = Field(
        ge=0,
        le=72,
    )

    phoneservice: Literal[
        "Yes",
        "No",
    ]

    multiplelines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    internetservice: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    onlinesecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    onlinebackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    deviceprotection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    techsupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    streamingtv: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    streamingmovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    paperlessbilling: Literal[
        "Yes",
        "No",
    ]

    paymentmethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    monthlycharges: float = Field(
        ge=0,
    )

    totalcharges: float = Field(
        ge=0,
    )