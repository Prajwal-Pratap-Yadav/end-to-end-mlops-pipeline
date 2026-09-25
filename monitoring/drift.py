"""Statistical data-drift detection between reference and production data.

For every model input (and for the model's output distribution) we compute:

* **PSI** (Population Stability Index) - the decision statistic. Numeric
  features are binned on reference quantiles; categorical features use their
  categories. PSI < 0.1 is conventionally stable, 0.1-0.2 moderate, >= 0.2 a
  significant shift. Unlike p-values it does not shrink towards "significant"
  as the window grows, which keeps alerting stable across traffic volumes.
* A **two-sample test** as supporting evidence: Kolmogorov-Smirnov for numeric
  features, chi-squared for categorical/binary ones.

A dataset is flagged as drifted when the share of drifted features reaches
``monitoring.drift.drift_share_threshold`` (15% by default, i.e. 2 of the 11
inputs - calibrated so same-distribution windows never alert while a real market
shift, which typically moves a few key inputs strongly, does).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DriftConfig
from src.schema import FEATURES

_EPSILON = 1e-4


@dataclass(frozen=True)
class FeatureDrift:
    """Drift statistics for one column."""

    feature: str
    kind: str
    psi: float
    test: str
    statistic: float
    p_value: float
    drift_detected: bool
    reference: dict[str, Any] = field(default_factory=dict)
    current: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Drift results for a production window versus the reference data."""

    features: list[FeatureDrift]
    n_reference: int
    n_current: int
    drift_share: float
    dataset_drift: bool
    prediction_drift: FeatureDrift | None
    thresholds: dict[str, float]

    @property
    def drifted_features(self) -> list[str]:
        """Names of features whose drift was detected."""
        return [f.feature for f in self.features if f.drift_detected]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reports."""
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "drift_share": self.drift_share,
            "dataset_drift": self.dataset_drift,
            "drifted_features": self.drifted_features,
            "thresholds": self.thresholds,
            "features": [asdict(f) for f in self.features],
            "prediction_drift": asdict(self.prediction_drift) if self.prediction_drift else None,
        }


def _proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / max(len(counts), 1))
    return np.asarray(np.clip(counts / total, _EPSILON, None), dtype=float)


def _psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    expected = _proportions(expected_counts.astype(float))
    actual = _proportions(actual_counts.astype(float))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """PSI of a numeric variable using quantile bins fitted on the reference.

    Args:
        reference: Reference sample (NaNs are ignored).
        current: Current sample (NaNs are ignored).
        n_bins: Number of quantile bins (fewer if the reference has ties).

    Returns:
        The population stability index (0 means identical binned distributions).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    n_buckets = len(edges) + 1
    ref_counts = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=n_buckets)
    cur_counts = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=n_buckets)
    return _psi_from_counts(ref_counts, cur_counts)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI of a categorical variable over the union of observed categories."""
    ref_counts = reference.dropna().astype(str).value_counts()
    cur_counts = current.dropna().astype(str).value_counts()
    if ref_counts.empty or cur_counts.empty:
        return float("nan")
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    return _psi_from_counts(
        ref_counts.reindex(categories, fill_value=0).to_numpy(),
        cur_counts.reindex(categories, fill_value=0).to_numpy(),
    )


def chi2_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Chi-squared test of homogeneity between two categorical samples."""
    ref_counts = reference.dropna().astype(str).value_counts()
    cur_counts = current.dropna().astype(str).value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    if len(categories) < 2 or ref_counts.empty or cur_counts.empty:
        return 0.0, 1.0
    table = np.vstack(
        [
            ref_counts.reindex(categories, fill_value=0).to_numpy(),
            cur_counts.reindex(categories, fill_value=0).to_numpy(),
        ]
    )
    result = stats.chi2_contingency(table)
    return float(result.statistic), float(result.pvalue)


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test (NaNs ignored)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0, 1.0
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def _numeric_summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "mean": float(numeric.mean()) if numeric.notna().any() else None,
        "std": float(numeric.std()) if numeric.notna().sum() > 1 else None,
        "missing_fraction": float(numeric.isna().mean()),
    }


def _categorical_summary(values: pd.Series) -> dict[str, Any]:
    shares = values.dropna().astype(str).value_counts(normalize=True).sort_index()
    return {"distribution": {k: round(float(v), 4) for k, v in shares.items()}}


def column_drift(
    name: str, kind: str, reference: pd.Series, current: pd.Series, config: DriftConfig
) -> FeatureDrift:
    """Compute PSI, the matching statistical test and summaries for one column."""
    if kind in ("numeric", "prediction"):
        ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
        cur = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
        psi = numeric_psi(ref, cur, config.n_bins)
        statistic, p_value = ks_test(ref, cur)
        test = "ks"
        ref_summary, cur_summary = _numeric_summary(reference), _numeric_summary(current)
    else:
        psi = categorical_psi(reference, current)
        statistic, p_value = chi2_test(reference, current)
        test = "chi2"
        ref_summary, cur_summary = _categorical_summary(reference), _categorical_summary(current)
    return FeatureDrift(
        feature=name,
        kind=kind,
        psi=psi,
        test=test,
        statistic=statistic,
        p_value=p_value,
        drift_detected=bool(np.isfinite(psi) and psi >= config.psi_threshold),
        reference=ref_summary,
        current=cur_summary,
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
    reference_predictions: pd.Series | None = None,
    current_predictions: pd.Series | None = None,
) -> DriftReport:
    """Compare a production window with the reference data, feature by feature.

    Args:
        reference: Reference feature table (the champion's training data).
        current: Recent production inputs.
        config: Drift thresholds.
        reference_predictions: Champion probabilities on held-out reference data.
        current_predictions: Probabilities served in the production window.

    Returns:
        A :class:`DriftReport`.
    """
    results = [
        column_drift(spec.name, spec.kind, reference[spec.name], current[spec.name], config)
        for spec in FEATURES
    ]
    prediction_drift = None
    if reference_predictions is not None and current_predictions is not None:
        prediction_drift = column_drift(
            "churn_probability", "prediction", reference_predictions, current_predictions, config
        )
    drift_share = sum(r.drift_detected for r in results) / len(results)
    return DriftReport(
        features=results,
        n_reference=len(reference),
        n_current=len(current),
        drift_share=drift_share,
        dataset_drift=drift_share >= config.drift_share_threshold,
        prediction_drift=prediction_drift,
        thresholds={
            "psi_threshold": config.psi_threshold,
            "p_value_threshold": config.p_value_threshold,
            "drift_share_threshold": config.drift_share_threshold,
        },
    )
