"""Feature engineering and preprocessing, packaged as a single scikit-learn Pipeline.

Everything that transforms raw inputs lives inside the model artifact, so the
API, the batch scorer and the monitoring job apply exactly the transformations
used in training - there is no separate serving-side feature code to drift out of
sync.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import FEATURES_BY_NAME, TARGET_COLUMN, feature_names, features_of_kind
from src.validation import coerce_types

ENGINEERED_NUMERIC = ["tickets_per_tenure_year", "charge_to_market_ratio"]
ENGINEERED_BINARY = ["is_new_customer"]


class ChurnFeatureEngineer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Domain-driven feature engineering applied as the first pipeline step.

    * ``total_charges`` is missing only for customers who have not been billed yet
      (tenure 0); it is imputed from tenure x monthly charges rather than a median.
    * ``tickets_per_tenure_year`` - support intensity normalised by relationship length.
    * ``charge_to_market_ratio`` - the customer's bill relative to the training-time
      median bill (captures price sensitivity even as absolute prices move).
    * ``is_new_customer`` - tenure under six months.

    The class is allow-listed for skops serialization when the model is logged, so
    the artifact can be loaded without unpickling arbitrary code.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> ChurnFeatureEngineer:
        """Learn the median monthly charge used as the market reference price."""
        charges = pd.to_numeric(X["monthly_charges"], errors="coerce")
        self.market_price_ = float(np.nanmedian(charges.to_numpy(dtype=float)))
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered columns to a copy of ``X``."""
        out = coerce_types(pd.DataFrame(X))
        tenure = out["tenure_months"]
        out["total_charges"] = out["total_charges"].fillna(tenure * out["monthly_charges"])
        out["tickets_per_tenure_year"] = out["num_support_tickets"] / (tenure / 12.0 + 1.0)
        out["charge_to_market_ratio"] = out["monthly_charges"] / max(self.market_price_, 1e-6)
        out["is_new_customer"] = (tenure < 6).astype(float)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output column names (inputs plus engineered features)."""
        names = list(self.feature_names_in_) + ENGINEERED_NUMERIC + ENGINEERED_BINARY
        return np.asarray(names, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Column-wise imputation, scaling and one-hot encoding."""
    numeric = features_of_kind("numeric") + ENGINEERED_NUMERIC
    binary = features_of_kind("binary") + ENGINEERED_BINARY
    categorical = features_of_kind("categorical")
    categories = [list(FEATURES_BY_NAME[name].categories) for name in categorical]

    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    binary_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("binary", binary_pipeline, binary),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assemble feature engineering, preprocessing and the estimator into one Pipeline."""
    return Pipeline(
        [
            ("features", ChurnFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled table into model inputs (contract features only) and target."""
    X = coerce_types(frame[feature_names()])
    y = frame[TARGET_COLUMN].astype(int)
    return X, y


def split_data(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of a labeled table.

    Args:
        frame: Labeled customer table.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``(train_frame, test_frame)`` with all original columns preserved.
    """
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
