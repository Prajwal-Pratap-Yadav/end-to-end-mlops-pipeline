from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig
from src.ingest import (
    DriftProfile,
    _shift_probabilities,
    generate_customers,
    ingest,
    load_csv,
    load_source_data,
    main,
)
from src.schema import ID_COLUMN, TARGET_COLUMN, feature_names
from src.validation import validate_dataset


def test_generator_is_deterministic_and_matches_contract() -> None:
    first = generate_customers(2000, seed=3)
    second = generate_customers(2000, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [ID_COLUMN, *feature_names(), TARGET_COLUMN]
    assert validate_dataset(first, min_rows=100).passed


def test_baseline_population_is_realistic() -> None:
    customers = generate_customers(8000, seed=1)
    assert 0.20 < customers[TARGET_COLUMN].mean() < 0.32  # telecom churn is ~26%
    # total_charges is missing exactly for customers who have not been billed yet.
    missing = customers["total_charges"].isna()
    assert (customers.loc[missing, "tenure_months"] == 0).all()
    assert customers.loc[customers["internet_service"] == "none", "has_tech_support"].eq(0).all()
    # Month-to-month customers churn more than two-year customers.
    rates = customers.groupby("contract_type")[TARGET_COLUMN].mean()
    assert rates["month_to_month"] > 2 * rates["two_year"]


def test_drift_moves_inputs_and_outcomes() -> None:
    base = generate_customers(6000, seed=5)
    shifted = generate_customers(6000, seed=5, drift=DriftProfile.from_strength(1.0))
    assert shifted["monthly_charges"].mean() > 1.2 * base["monthly_charges"].mean()
    assert shifted["num_support_tickets"].mean() > base["num_support_tickets"].mean() + 1
    assert (shifted["contract_type"] == "month_to_month").mean() > (
        base["contract_type"] == "month_to_month"
    ).mean()
    assert shifted[TARGET_COLUMN].mean() > base[TARGET_COLUMN].mean()
    assert validate_dataset(shifted).passed  # drifted data is still contract-valid


def test_drift_strength_is_clipped() -> None:
    assert DriftProfile.from_strength(5.0) == DriftProfile.from_strength(1.0)
    assert DriftProfile.from_strength(-1.0) == DriftProfile()


def test_shift_probabilities_moves_mass_and_stays_normalised() -> None:
    probs = _shift_probabilities(np.array([0.5, 0.3, 0.2]), target_index=0, mass=0.2)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.7)
    assert probs[1] / probs[2] == pytest.approx(0.3 / 0.2)


def test_invalid_sample_count_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_customers(0)


def test_ingest_writes_snapshot(config: AppConfig) -> None:
    frame, path = ingest(config)
    assert path == config.data.raw_path
    assert len(pd.read_csv(path)) == len(frame) == config.data.n_samples


def test_csv_source_round_trip(config: AppConfig, tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    generate_customers(300, seed=9).to_csv(csv_path, index=False)
    csv_config = config.model_copy(
        update={"data": config.data.model_copy(update={"source": "csv", "csv_path": csv_path})}
    )
    assert len(load_source_data(csv_config)) == 300
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_cli_generates_drifted_file(tmp_path: Path) -> None:
    output = tmp_path / "drifted.csv"
    main(["--n-samples", "250", "--drift-strength", "0.8", "--seed", "4", "--output", str(output)])
    frame = pd.read_csv(output)
    assert len(frame) == 250
    assert validate_dataset(frame).passed
