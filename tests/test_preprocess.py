import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.ingest import generate_customers
from src.preprocess import (
    ENGINEERED_BINARY,
    ENGINEERED_NUMERIC,
    ChurnFeatureEngineer,
    build_pipeline,
    split_data,
    split_features_target,
)
from src.schema import TARGET_COLUMN, feature_names


@pytest.fixture
def customers() -> pd.DataFrame:
    return generate_customers(1200, seed=4)


def test_feature_engineer_adds_domain_features(customers: pd.DataFrame) -> None:
    X, _ = split_features_target(customers)
    engineer = ChurnFeatureEngineer().fit(X)
    out = engineer.transform(X)
    for column in ENGINEERED_NUMERIC + ENGINEERED_BINARY:
        assert column in out.columns
    assert engineer.market_price_ == pytest.approx(X["monthly_charges"].median())
    assert list(engineer.get_feature_names_out()) == list(out.columns)
    assert out["is_new_customer"].isin([0.0, 1.0]).all()


def test_unbilled_customers_get_zero_total_charges(customers: pd.DataFrame) -> None:
    X, _ = split_features_target(customers)
    X.loc[X.index[0], ["tenure_months", "total_charges"]] = [0, np.nan]
    out = ChurnFeatureEngineer().fit(X).transform(X)
    assert out.loc[X.index[0], "total_charges"] == 0.0
    assert out["total_charges"].notna().all()


def test_pipeline_trains_and_handles_missing_and_unseen_values(customers: pd.DataFrame) -> None:
    X, y = split_features_target(customers)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)

    tricky = X.head(3).copy()
    tricky.loc[tricky.index[0], "avg_monthly_gb"] = np.nan
    tricky.loc[tricky.index[1], "payment_method"] = "crypto"  # never seen in training
    proba = pipeline.predict_proba(tricky)
    assert proba.shape == (3, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_pipeline_ignores_extra_columns(customers: pd.DataFrame) -> None:
    X, y = split_features_target(customers)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    with_extra = X.head(5).assign(customer_note="vip")
    np.testing.assert_allclose(
        pipeline.predict_proba(with_extra[feature_names()]), pipeline.predict_proba(X.head(5))
    )


def test_split_is_stratified_and_disjoint(customers: pd.DataFrame) -> None:
    train, test = split_data(customers, test_size=0.25, random_state=0)
    assert len(test) == 300
    assert len(train) == 900
    assert abs(train[TARGET_COLUMN].mean() - test[TARGET_COLUMN].mean()) < 0.02
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])
