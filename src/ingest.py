"""Data ingestion: produce an immutable raw snapshot for a training run.

Two sources are supported (``data.source`` in the config):

* ``synthetic`` - a documented data-generating process for a subscription
  business's customer churn. It is reproducible, needs no network or licence, and
  can inject *covariate drift* (input distributions move) and *concept drift*
  (the relationship between inputs and churn changes). That makes the monitoring
  and retraining loop demonstrable end to end.
* ``csv`` - any CSV that satisfies the feature contract in :mod:`src.schema`.

Usage:
    python -m src.ingest                       # snapshot per configs/config.yaml
    python -m src.ingest --n-samples 2000 --drift-strength 0.8 --output data/raw/drifted.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.schema import ID_COLUMN, TARGET_COLUMN, feature_names
from src.utils import configure_logging, ensure_parent_dir

logger = logging.getLogger(__name__)

CONTRACTS = np.array(["month_to_month", "one_year", "two_year"])
INTERNET = np.array(["dsl", "fiber_optic", "none"])
PAYMENTS = np.array(["electronic_check", "mailed_check", "bank_transfer", "credit_card"])

_BASE_CONTRACT_P = np.array([0.55, 0.24, 0.21])
_BASE_INTERNET_P = np.array([0.34, 0.44, 0.22])
_PAYMENT_P = np.array([0.34, 0.23, 0.22, 0.21])


@dataclass(frozen=True)
class DriftProfile:
    """Parameters that move the synthetic population away from the baseline.

    Covariate drift (what customers look like):
        price_increase: Relative increase in monthly charges (0.3 = +30%).
        month_to_month_shift: Probability mass moved onto month-to-month contracts.
        fiber_shift: Probability mass moved onto fiber-optic internet.
        support_ticket_increase: Added to the Poisson rate of support tickets.

    Concept drift (how customers behave):
        concept_shift: 0 keeps the baseline churn mechanism; 1 is a market where a
            competitor sells cheap fiber - customers become far more price-sensitive,
            DSL customers defect, and payment method stops mattering.
    """

    price_increase: float = 0.0
    month_to_month_shift: float = 0.0
    fiber_shift: float = 0.0
    support_ticket_increase: float = 0.0
    concept_shift: float = 0.0

    @classmethod
    def from_strength(cls, strength: float) -> DriftProfile:
        """Build a combined covariate + concept drift profile from a scalar in [0, 1]."""
        s = float(np.clip(strength, 0.0, 1.0))
        return cls(
            price_increase=0.30 * s,
            month_to_month_shift=0.20 * s,
            fiber_shift=0.15 * s,
            support_ticket_increase=1.5 * s,
            concept_shift=s,
        )


def _shift_probabilities(base: np.ndarray, target_index: int, mass: float) -> np.ndarray:
    """Move ``mass`` of probability onto ``target_index`` proportionally from the others."""
    probs = base.astype(float).copy()
    others = [i for i in range(len(probs)) if i != target_index]
    available = probs[others].sum()
    moved = min(max(mass, 0.0), available)
    probs[others] -= moved * probs[others] / available
    probs[target_index] += moved
    return np.asarray(probs / probs.sum(), dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=float)


def generate_customers(
    n_samples: int,
    seed: int | None = None,
    drift: DriftProfile | None = None,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic customer table with a ``churned`` label.

    Args:
        n_samples: Number of customers to generate.
        seed: Seed for the random generator (``None`` for non-deterministic output).
        drift: Optional drift profile; defaults to the baseline population.
        id_offset: Offset for ``customer_id`` numbering so batches do not collide.

    Returns:
        DataFrame with ``customer_id``, every feature in :mod:`src.schema` and ``churned``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    d = drift or DriftProfile()
    c = float(np.clip(d.concept_shift, 0.0, 1.0))

    contract = rng.choice(
        CONTRACTS,
        size=n_samples,
        p=_shift_probabilities(_BASE_CONTRACT_P, 0, d.month_to_month_shift),
    )
    is_m2m = contract == "month_to_month"
    tenure = np.where(
        is_m2m,
        rng.gamma(shape=1.2, scale=14.0, size=n_samples),
        np.where(
            contract == "one_year",
            rng.uniform(6, 72, size=n_samples),
            rng.uniform(12, 72, size=n_samples),
        ),
    )
    tenure = np.clip(np.round(tenure), 0, 72).astype(int)

    internet = rng.choice(
        INTERNET, size=n_samples, p=_shift_probabilities(_BASE_INTERNET_P, 1, d.fiber_shift)
    )
    has_internet = internet != "none"
    has_tech_support = (has_internet & (rng.random(n_samples) < 0.35)).astype(int)

    base_charge = np.select(
        [internet == "dsl", internet == "fiber_optic"],
        [rng.normal(55, 8, n_samples), rng.normal(85, 9, n_samples)],
        default=rng.normal(22, 2, n_samples),
    )
    monthly_charges = np.clip(
        (base_charge + 10 * has_tech_support) * (1 + d.price_increase), 18, 250
    )
    monthly_charges = np.round(monthly_charges, 2)

    total_charges = np.round(tenure * monthly_charges * rng.uniform(0.92, 1.05, n_samples), 2)
    total_charges = np.where(tenure == 0, np.nan, total_charges)

    payment = rng.choice(PAYMENTS, size=n_samples, p=_PAYMENT_P)
    paperless = (rng.random(n_samples) < 0.60).astype(int)
    senior = (rng.random(n_samples) < 0.16).astype(int)
    tickets = rng.poisson(
        0.8
        + 0.7 * (internet == "fiber_optic")
        + 0.5 * (tenure < 6)
        + max(d.support_ticket_increase, 0.0)
    )

    usage = np.where(
        internet == "fiber_optic",
        rng.lognormal(np.log(90), 0.5, n_samples),
        rng.lognormal(np.log(45), 0.5, n_samples),
    )
    usage = np.round(np.where(has_internet, usage, 0.0), 1)
    usage = np.where(rng.random(n_samples) < 0.03, np.nan, usage)  # telemetry gaps

    # Churn mechanism. Each coefficient interpolates between the baseline market (c=0)
    # and the post-competitor market (c=1): loyalty from tenure erodes, customers react
    # to price relative to the market, fiber becomes the sticky product, DSL customers
    # defect and the payment-method signal disappears. Risk factors reorder, so a model
    # trained on the old market ranks customers worse - which is what retraining fixes.
    market_price = 65.0 * (1 + d.price_increase)
    logit = (
        -2.05
        - 0.55 * c
        + 1.30 * is_m2m
        - 0.90 * (contract == "two_year")
        - (0.035 - 0.035 * c) * tenure
        + (0.020 + 0.040 * c) * (monthly_charges - market_price)
        + 0.35 * tickets
        + (0.55 - 0.55 * c) * (payment == "electronic_check")
        + (0.45 - 1.25 * c) * (internet == "fiber_optic")
        + (0.00 + 1.20 * c) * (internet == "dsl")
        - 0.50 * has_tech_support
        + 0.30 * senior
        + 0.20 * paperless
        + rng.normal(0, 0.5, n_samples)
    )
    churned = (rng.random(n_samples) < _sigmoid(logit)).astype(int)

    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"C{id_offset + i:07d}" for i in range(n_samples)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": tickets.astype(int),
            "avg_monthly_gb": usage,
            "senior_citizen": senior,
            "paperless_billing": paperless,
            "has_tech_support": has_tech_support,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            TARGET_COLUMN: churned,
        }
    )
    return frame[[ID_COLUMN, *feature_names(), TARGET_COLUMN]]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a customer table from CSV."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_source_data(config: AppConfig, drift: DriftProfile | None = None) -> pd.DataFrame:
    """Load raw training data from the configured source.

    Args:
        config: Application configuration.
        drift: Drift profile for the synthetic source (ignored for CSV sources).

    Returns:
        Raw customer table.
    """
    if config.data.source == "csv":
        if config.data.csv_path is None:  # also enforced by DataConfig validation
            raise ValueError("data.csv_path is required when data.source == 'csv'")
        logger.info("Loading CSV source %s", config.data.csv_path)
        return load_csv(config.data.csv_path)
    logger.info("Generating %d synthetic customers", config.data.n_samples)
    return generate_customers(config.data.n_samples, seed=config.project.random_seed, drift=drift)


def ingest(config: AppConfig, output_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Materialize the raw snapshot for a training run and optionally publish it to S3.

    Args:
        config: Application configuration.
        output_path: Where to write the snapshot; defaults to ``data.raw_path``.

    Returns:
        The raw DataFrame and the path of the written snapshot.
    """
    frame = load_source_data(config)
    target = ensure_parent_dir(output_path or config.data.raw_path)
    frame.to_csv(target, index=False)
    logger.info("Wrote raw snapshot: %s (%d rows)", target, len(frame))

    if config.storage.s3.enabled:
        from src.s3 import S3Storage  # imported lazily: boto3 is only needed when enabled

        storage = S3Storage.from_config(config.storage.s3)
        uri = storage.upload_file(target, f"data/raw/{target.name}")
        logger.info("Published raw snapshot to %s", uri)
    return frame, target


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--n-samples", type=int, default=None, help="Rows to generate (synthetic)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (synthetic)")
    parser.add_argument(
        "--drift-strength",
        type=float,
        default=0.0,
        help="Covariate + concept drift in [0, 1] (synthetic source only)",
    )
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    if args.drift_strength > 0 or args.n_samples or args.seed is not None:
        frame = generate_customers(
            args.n_samples or config.data.n_samples,
            seed=config.project.random_seed if args.seed is None else args.seed,
            drift=DriftProfile.from_strength(args.drift_strength),
        )
        target = ensure_parent_dir(args.output or config.data.raw_path)
        frame.to_csv(target, index=False)
        logger.info("Wrote %d rows to %s", len(frame), target)
    else:
        _, target = ingest(config, args.output)
    print(target)


if __name__ == "__main__":
    main()
