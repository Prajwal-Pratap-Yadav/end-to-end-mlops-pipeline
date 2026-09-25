"""Data contract validation run before any model is trained.

Training on silently broken data is one of the most common production ML
failures, so the pipeline refuses to continue when the raw snapshot violates the
contract defined in :mod:`src.schema` (missing columns, wrong types, values out
of range, unknown categories, too many nulls, a degenerate target, ...).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

from src.schema import FEATURES, ID_COLUMN, TARGET_COLUMN, feature_names

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """A single contract violation."""

    column: str
    check: str
    message: str
    severity: Severity = "error"


@dataclass
class ValidationReport:
    """Outcome of validating a dataset against the feature contract."""

    n_rows: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues that must block the pipeline."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def passed(self) -> bool:
        """True when no blocking issues were found."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for logging as an MLflow artifact."""
        return {
            "passed": self.passed,
            "n_rows": self.n_rows,
            "n_errors": len(self.errors),
            "n_warnings": len(self.issues) - len(self.errors),
            "issues": [asdict(issue) for issue in self.issues],
        }


class DataValidationError(ValueError):
    """Raised when a dataset violates the data contract."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        details = "; ".join(f"[{i.column}] {i.message}" for i in report.errors[:10])
        super().__init__(f"Data validation failed with {len(report.errors)} error(s): {details}")


def validate_dataset(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    min_rows: int = 1,
    max_missing_fraction: float = 0.2,
) -> ValidationReport:
    """Validate a customer table against the feature contract.

    Args:
        frame: Data to validate.
        require_target: Whether the ``churned`` label must be present and valid.
        min_rows: Minimum number of rows required.
        max_missing_fraction: Maximum tolerated null fraction for nullable features.

    Returns:
        A :class:`ValidationReport`; check ``report.passed``.
    """
    report = ValidationReport(n_rows=len(frame))
    issues = report.issues

    if len(frame) < min_rows:
        issues.append(
            ValidationIssue("*", "min_rows", f"{len(frame)} rows < required minimum {min_rows}")
        )

    required = feature_names() + ([TARGET_COLUMN] if require_target else [])
    missing = [col for col in required if col not in frame.columns]
    for col in missing:
        issues.append(ValidationIssue(col, "required_column", "column is missing"))

    for spec in FEATURES:
        if spec.name in missing:
            continue
        column = frame[spec.name]
        null_fraction = float(column.isna().mean()) if len(column) else 0.0
        if not spec.nullable and null_fraction > 0:
            issues.append(
                ValidationIssue(
                    spec.name, "not_null", f"{null_fraction:.1%} missing in a non-nullable column"
                )
            )
        elif null_fraction > max_missing_fraction:
            issues.append(
                ValidationIssue(
                    spec.name,
                    "missing_fraction",
                    f"{null_fraction:.1%} missing exceeds budget of {max_missing_fraction:.0%}",
                )
            )

        present = column.dropna()
        if spec.kind in ("numeric", "binary"):
            numeric = pd.to_numeric(present, errors="coerce")
            n_bad_type = int(numeric.isna().sum())
            if n_bad_type:
                issues.append(
                    ValidationIssue(spec.name, "dtype", f"{n_bad_type} non-numeric value(s)")
                )
            numeric = numeric.dropna()
            if spec.kind == "binary":
                n_invalid = int((~numeric.isin([0, 1])).sum())
                if n_invalid:
                    issues.append(
                        ValidationIssue(
                            spec.name, "binary", f"{n_invalid} value(s) not in {{0, 1}}"
                        )
                    )
            else:
                low = spec.min_value if spec.min_value is not None else float("-inf")
                high = spec.max_value if spec.max_value is not None else float("inf")
                n_out = int(((numeric < low) | (numeric > high)).sum())
                if n_out:
                    issues.append(
                        ValidationIssue(
                            spec.name, "range", f"{n_out} value(s) outside [{low}, {high}]"
                        )
                    )
        else:
            unknown = sorted(set(present.astype(str)) - set(spec.categories))
            if unknown:
                issues.append(
                    ValidationIssue(spec.name, "categories", f"unknown categories: {unknown[:5]}")
                )

    if require_target and TARGET_COLUMN not in missing:
        target = frame[TARGET_COLUMN]
        if target.isna().any():
            issues.append(ValidationIssue(TARGET_COLUMN, "not_null", "target has missing values"))
        values = set(pd.to_numeric(target.dropna(), errors="coerce").unique())
        if not values <= {0, 1}:
            issues.append(ValidationIssue(TARGET_COLUMN, "binary", f"target values {values}"))
        elif len(values) < 2:
            issues.append(
                ValidationIssue(TARGET_COLUMN, "class_balance", "target has a single class")
            )

    if ID_COLUMN in frame.columns:
        n_dupes = int(frame[ID_COLUMN].duplicated().sum())
        if n_dupes:
            issues.append(
                ValidationIssue(
                    ID_COLUMN, "unique", f"{n_dupes} duplicated id(s)", severity="warning"
                )
            )

    for issue in issues:
        log = logger.error if issue.severity == "error" else logger.warning
        log("Validation %s [%s] %s", issue.check, issue.column, issue.message)
    return report


def assert_valid(frame: pd.DataFrame, **kwargs: Any) -> ValidationReport:
    """Validate ``frame`` and raise :class:`DataValidationError` on any blocking issue.

    Args:
        frame: Data to validate.
        **kwargs: Forwarded to :func:`validate_dataset`.

    Returns:
        The (passing) validation report.
    """
    report = validate_dataset(frame, **kwargs)
    if not report.passed:
        raise DataValidationError(report)
    return report


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric/binary features as floats and categoricals as strings.

    Applied after validation so the model always sees the same dtypes, whether the
    data came from a CSV, the synthetic generator, the API or the prediction log.
    """
    result = frame.copy()
    for spec in FEATURES:
        if spec.name not in result.columns:
            continue
        if spec.kind in ("numeric", "binary"):
            result[spec.name] = pd.to_numeric(result[spec.name], errors="coerce").astype(float)
        else:
            result[spec.name] = result[spec.name].astype(object).where(result[spec.name].notna())
    return result
