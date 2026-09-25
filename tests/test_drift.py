import json
import math

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import categorical_psi, chi2_test, detect_drift, ks_test, numeric_psi
from src.config import DriftConfig
from src.ingest import DriftProfile, generate_customers

rng = np.random.default_rng(0)


def test_psi_is_near_zero_for_same_distribution() -> None:
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(0, 1, 2000)
    assert numeric_psi(reference, current) < 0.02


def test_psi_grows_with_the_size_of_the_shift() -> None:
    reference = rng.normal(0, 1, 5000)
    small = numeric_psi(reference, rng.normal(0.2, 1, 2000))
    large = numeric_psi(reference, rng.normal(1.0, 1, 2000))
    assert 0.02 < small < 0.2 < large


def test_psi_ignores_nans_and_handles_empty_input() -> None:
    reference = np.array([1.0, 2.0, np.nan, 3.0, 4.0] * 100)
    assert numeric_psi(reference, reference) == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(numeric_psi(reference, np.array([np.nan])))


def test_psi_handles_heavily_tied_integer_features() -> None:
    reference = rng.poisson(1.0, 5000)
    assert numeric_psi(reference, rng.poisson(1.0, 2000)) < 0.02
    assert numeric_psi(reference, rng.poisson(2.5, 2000)) > 0.2


def test_categorical_psi_and_new_categories() -> None:
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    assert categorical_psi(reference, reference.sample(frac=1, random_state=0)) < 1e-9
    assert categorical_psi(reference, pd.Series(["a"] * 900 + ["b"] * 100)) > 0.2
    assert categorical_psi(reference, pd.Series(["a"] * 500 + ["c"] * 500)) > 1.0


def test_statistical_tests() -> None:
    statistic, p_value = ks_test(rng.normal(0, 1, 500), rng.normal(2, 1, 500))
    assert statistic > 0.5
    assert p_value < 1e-6
    same = pd.Series(["x", "y"] * 200)
    assert chi2_test(same, same)[1] == pytest.approx(1.0)
    assert chi2_test(pd.Series(["x"] * 10), pd.Series(["x"] * 10)) == (0.0, 1.0)


def test_no_false_alarm_on_fresh_sample_of_training_distribution() -> None:
    reference = generate_customers(4000, seed=1)
    for seed in (2, 3, 4):
        report = detect_drift(reference, generate_customers(1000, seed=seed), DriftConfig())
        assert not report.dataset_drift
        assert report.drifted_features == []


def test_market_shift_is_detected_in_the_right_features() -> None:
    reference = generate_customers(4000, seed=1)
    current = generate_customers(1000, seed=2, drift=DriftProfile.from_strength(0.8))
    report = detect_drift(reference, current, DriftConfig())
    assert report.dataset_drift
    assert {"monthly_charges", "num_support_tickets"} <= set(report.drifted_features)
    assert "senior_citizen" not in report.drifted_features  # untouched by the shift
    assert report.drift_share == pytest.approx(len(report.drifted_features) / 11)


def test_prediction_drift_and_serialisation() -> None:
    reference = generate_customers(1000, seed=1)
    report = detect_drift(
        reference,
        generate_customers(500, seed=2),
        DriftConfig(),
        reference_predictions=pd.Series(rng.beta(2, 5, 1000)),
        current_predictions=pd.Series(rng.beta(5, 2, 500)),
    )
    assert report.prediction_drift is not None
    assert report.prediction_drift.drift_detected
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["prediction_drift"]["feature"] == "churn_probability"
    assert len(payload["features"]) == 11
