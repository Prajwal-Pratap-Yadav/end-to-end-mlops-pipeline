import numpy as np
import pandas as pd
import pytest

from src.ingest import generate_customers
from src.schema import TARGET_COLUMN
from src.validation import DataValidationError, assert_valid, coerce_types, validate_dataset


@pytest.fixture
def frame() -> pd.DataFrame:
    return generate_customers(400, seed=2)


def _checks(frame: pd.DataFrame, **kwargs: object) -> set[tuple[str, str]]:
    report = validate_dataset(frame, **kwargs)  # type: ignore[arg-type]
    return {(issue.column, issue.check) for issue in report.errors}


def test_clean_data_passes(frame: pd.DataFrame) -> None:
    report = validate_dataset(frame, min_rows=100)
    assert report.passed
    assert report.to_dict()["n_errors"] == 0


def test_missing_column_is_an_error(frame: pd.DataFrame) -> None:
    assert ("tenure_months", "required_column") in _checks(frame.drop(columns="tenure_months"))


def test_out_of_range_values_are_errors(frame: pd.DataFrame) -> None:
    frame.loc[0, "monthly_charges"] = -5
    frame.loc[1, "tenure_months"] = 500
    checks = _checks(frame)
    assert ("monthly_charges", "range") in checks
    assert ("tenure_months", "range") in checks


def test_unknown_category_is_an_error(frame: pd.DataFrame) -> None:
    frame.loc[0, "contract_type"] = "weekly"
    assert ("contract_type", "categories") in _checks(frame)


def test_non_binary_flag_is_an_error(frame: pd.DataFrame) -> None:
    frame.loc[0, "senior_citizen"] = 2
    assert ("senior_citizen", "binary") in _checks(frame)


def test_non_numeric_value_is_an_error(frame: pd.DataFrame) -> None:
    frame["num_support_tickets"] = frame["num_support_tickets"].astype(object)
    frame.loc[0, "num_support_tickets"] = "many"
    assert ("num_support_tickets", "dtype") in _checks(frame)


def test_nulls_in_non_nullable_column_are_errors(frame: pd.DataFrame) -> None:
    frame.loc[0, "payment_method"] = None
    assert ("payment_method", "not_null") in _checks(frame)


def test_missing_budget_applies_to_nullable_columns(frame: pd.DataFrame) -> None:
    frame.loc[: len(frame) // 2, "avg_monthly_gb"] = np.nan
    assert ("avg_monthly_gb", "missing_fraction") in _checks(frame, max_missing_fraction=0.2)


def test_target_must_have_both_classes(frame: pd.DataFrame) -> None:
    frame[TARGET_COLUMN] = 0
    assert (TARGET_COLUMN, "class_balance") in _checks(frame)


def test_target_must_be_binary(frame: pd.DataFrame) -> None:
    frame.loc[0, TARGET_COLUMN] = 3
    assert (TARGET_COLUMN, "binary") in _checks(frame)


def test_target_is_optional_for_inference(frame: pd.DataFrame) -> None:
    assert validate_dataset(frame.drop(columns=TARGET_COLUMN), require_target=False).passed


def test_minimum_rows(frame: pd.DataFrame) -> None:
    assert ("*", "min_rows") in _checks(frame.head(5), min_rows=10)


def test_duplicate_ids_are_only_a_warning(frame: pd.DataFrame) -> None:
    duplicated = pd.concat([frame, frame.head(3)], ignore_index=True)
    report = validate_dataset(duplicated)
    assert report.passed
    assert [issue.check for issue in report.issues] == ["unique"]


def test_assert_valid_raises_with_details(frame: pd.DataFrame) -> None:
    frame.loc[0, "contract_type"] = "weekly"
    with pytest.raises(DataValidationError, match="unknown categories") as excinfo:
        assert_valid(frame)
    assert not excinfo.value.report.passed


def test_coerce_types_normalises_dtypes(frame: pd.DataFrame) -> None:
    raw = frame.astype({"tenure_months": str, "senior_citizen": str})
    coerced = coerce_types(raw)
    assert coerced["tenure_months"].dtype == float
    assert coerced["senior_citizen"].dtype == float
    assert coerced["contract_type"].dtype == object
    assert raw["tenure_months"].dtype != float  # input is not mutated
