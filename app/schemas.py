"""Request and response models for the inference API.

Field bounds, descriptions and allowed categories come from :mod:`src.schema`, so
the API contract cannot silently diverge from the data contract the model was
trained and validated against.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schema import FEATURES_BY_NAME, ContractType, InternetService, PaymentMethod


def _field(name: str, **kwargs: Any) -> Any:
    """Build a pydantic ``Field`` whose bounds and description come from the contract."""
    spec = FEATURES_BY_NAME[name]
    if spec.min_value is not None:
        kwargs.setdefault("ge", spec.min_value)
    if spec.max_value is not None:
        kwargs.setdefault("le", spec.max_value)
    return Field(description=spec.description, **kwargs)


EXAMPLE_CUSTOMER: dict[str, Any] = {
    "tenure_months": 4,
    "monthly_charges": 89.5,
    "total_charges": 358.0,
    "num_support_tickets": 3,
    "avg_monthly_gb": 120.4,
    "senior_citizen": False,
    "paperless_billing": True,
    "has_tech_support": False,
    "contract_type": "month_to_month",
    "internet_service": "fiber_optic",
    "payment_method": "electronic_check",
}


class CustomerFeatures(BaseModel):
    """Model inputs for one customer."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    tenure_months: int = _field("tenure_months")
    monthly_charges: float = _field("monthly_charges")
    total_charges: float | None = _field("total_charges", default=None)
    num_support_tickets: int = _field("num_support_tickets")
    avg_monthly_gb: float | None = _field("avg_monthly_gb", default=None)
    senior_citizen: bool = _field("senior_citizen")
    paperless_billing: bool = _field("paperless_billing")
    has_tech_support: bool = _field("has_tech_support")
    contract_type: ContractType = _field("contract_type")
    internet_service: InternetService = _field("internet_service")
    payment_method: PaymentMethod = _field("payment_method")

    def to_feature_dict(self) -> dict[str, Any]:
        """Return the features in the dtypes the model was trained on (booleans as 0/1)."""
        values = self.model_dump()
        for name, value in values.items():
            if isinstance(value, bool):
                values[name] = int(value)
        return values


class BatchPredictionRequest(BaseModel):
    """Several customers scored in one call."""

    model_config = ConfigDict(extra="forbid")

    instances: list[CustomerFeatures] = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Churn prediction for one customer."""

    prediction_id: str = Field(description="Use this id to submit ground truth to /feedback")
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: bool
    threshold: float
    model_name: str | None
    model_version: str | None


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch, in request order."""

    predictions: list[PredictionResponse]


class FeedbackRequest(BaseModel):
    """Ground truth observed after a prediction was served."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(min_length=1, max_length=64)
    churned: bool


class FeedbackResponse(BaseModel):
    """Acknowledgement that the outcome was recorded."""

    prediction_id: str
    status: Literal["recorded"] = "recorded"


class ModelInfoResponse(BaseModel):
    """Lineage of the model currently serving traffic."""

    name: str | None
    version: str | None
    run_id: str | None
    model_uri: str
    loaded_at: datetime
    decision_threshold: float
    training_metrics: dict[str, float]


class HealthResponse(BaseModel):
    """Liveness: the process is up."""

    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    """Readiness: a model is loaded and traffic can be served."""

    status: Literal["ready", "not_ready"]
    model_version: str | None = None
    detail: str | None = None


class ReloadResponse(BaseModel):
    """Result of a forced champion reload."""

    reloaded: bool
    previous_version: str | None
    current_version: str | None
