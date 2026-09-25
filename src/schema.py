"""Feature contract shared by ingestion, validation, training, serving and monitoring.

Keeping the schema in one place is what prevents training/serving skew: the API
request model, the data validator, the preprocessing pipeline and the drift
detector all derive their column lists, ranges and category domains from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

ContractType = Literal["month_to_month", "one_year", "two_year"]
InternetService = Literal["dsl", "fiber_optic", "none"]
PaymentMethod = Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]

FeatureKind = Literal["numeric", "binary", "categorical"]

ID_COLUMN = "customer_id"
TARGET_COLUMN = "churned"


@dataclass(frozen=True)
class FeatureSpec:
    """Contract for a single model input feature.

    Attributes:
        name: Column name.
        kind: ``numeric``, ``binary`` (0/1) or ``categorical``.
        description: Human-readable meaning, surfaced in the API docs.
        min_value: Inclusive lower bound for numeric features.
        max_value: Inclusive upper bound for numeric features.
        categories: Allowed values for categorical features.
        nullable: Whether missing values are legitimate (and imputed by the pipeline).
    """

    name: str
    kind: FeatureKind
    description: str
    min_value: float | None = None
    max_value: float | None = None
    categories: tuple[str, ...] = ()
    nullable: bool = False


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("tenure_months", "numeric", "Months the customer has been subscribed", 0, 120),
    FeatureSpec("monthly_charges", "numeric", "Current monthly bill in USD", 0, 500),
    FeatureSpec(
        "total_charges",
        "numeric",
        "Lifetime billed amount in USD (missing for brand-new customers)",
        0,
        60_000,
        nullable=True,
    ),
    FeatureSpec(
        "num_support_tickets", "numeric", "Support tickets opened in the last 90 days", 0, 100
    ),
    FeatureSpec(
        "avg_monthly_gb",
        "numeric",
        "Average monthly data usage in GB (missing when telemetry is unavailable)",
        0,
        5_000,
        nullable=True,
    ),
    FeatureSpec("senior_citizen", "binary", "1 if the customer is 65 or older"),
    FeatureSpec("paperless_billing", "binary", "1 if the customer uses paperless billing"),
    FeatureSpec("has_tech_support", "binary", "1 if the customer pays for tech support"),
    FeatureSpec(
        "contract_type", "categorical", "Contract commitment", categories=get_args(ContractType)
    ),
    FeatureSpec(
        "internet_service",
        "categorical",
        "Internet service tier",
        categories=get_args(InternetService),
    ),
    FeatureSpec(
        "payment_method", "categorical", "Payment method", categories=get_args(PaymentMethod)
    ),
)

FEATURES_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURES}


def feature_names() -> list[str]:
    """Return all model input feature names in canonical order."""
    return [spec.name for spec in FEATURES]


def features_of_kind(kind: FeatureKind) -> list[str]:
    """Return the names of all features of the given kind."""
    return [spec.name for spec in FEATURES if spec.kind == kind]
