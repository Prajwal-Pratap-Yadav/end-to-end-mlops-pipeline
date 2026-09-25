"""Training pipeline: data -> validation -> model selection -> evaluation -> registry.

Steps, all tracked in MLflow:

1. **Ingest** a raw snapshot (or use a DataFrame handed in by the retraining job).
2. **Validate** it against the data contract; abort on violations.
3. **Split** into train/evaluation sets (stratified; retraining passes a time-based holdout).
4. **Select a model**: every candidate/hyper-parameter combination is cross-validated
   and logged as a nested run; the best mean CV score on ``training.selection_metric`` wins.
5. **Fit and evaluate** the winner on held-out data: metrics, plots, permutation importance.
6. **Log the model** (skops-serialized, with signature, input example and pinned
   requirements) together with the reference data the monitor uses for drift detection.
7. **Register** it as a new version, tag it ``@challenger`` and run the promotion
   gate against the current ``@champion`` on identical evaluation data.

Usage:
    python -m src.train
    python -m src.train --skip-if-champion-exists   # idempotent bootstrap (docker compose)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import skops
from mlflow.data.pandas_dataset import from_pandas
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import AppConfig, ModelCandidate, load_config
from src.evaluate import compute_metrics, evaluate_model
from src.ingest import ingest
from src.predict import load_model
from src.preprocess import build_pipeline, split_data, split_features_target
from src.registry import (
    PromotionDecision,
    configure_mlflow,
    decide_promotion,
    get_version_by_alias,
    record_decision,
    register_model,
    set_alias,
)
from src.schema import TARGET_COLUMN, feature_names
from src.utils import configure_logging, write_json
from src.validation import assert_valid, coerce_types

logger = logging.getLogger(__name__)

# Explicit skops allow-list: loading a model can only instantiate these types (plus
# skops' built-in safe defaults), unlike pickle which can execute arbitrary code.
# Found with skops.io.get_untrusted_types() for every supported estimator.
TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "src.preprocess.ChurnFeatureEngineer",
]
CV_SCORING = ["roc_auc", "f1", "average_precision", "accuracy", "precision", "recall"]


@dataclass
class CandidateResult:
    """Cross-validation outcome for one candidate/hyper-parameter combination."""

    name: str
    estimator: str
    params: dict[str, Any]
    cv_mean: dict[str, float]
    cv_std: dict[str, float]
    run_id: str


@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    run_id: str
    model_name: str
    model_version: str
    best_candidate: str
    best_params: dict[str, Any]
    test_metrics: dict[str, float]
    decision: PromotionDecision
    champion_version: str | None
    candidates: list[CandidateResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary for logs and CLIs."""
        return {
            "run_id": self.run_id,
            "model": f"{self.model_name} v{self.model_version}",
            "best_candidate": self.best_candidate,
            "best_params": self.best_params,
            "test_metrics": {k: round(v, 4) for k, v in self.test_metrics.items()},
            "promoted": self.decision.promote,
            "decision_reason": self.decision.reason,
            "champion_version": self.champion_version,
        }


def build_estimator(name: str, params: dict[str, Any], seed: int) -> ClassifierMixin:
    """Instantiate a supported estimator with hyper-parameters.

    Args:
        name: One of ``logistic_regression``, ``random_forest``, ``hist_gradient_boosting``.
        params: Hyper-parameters passed to the constructor.
        seed: Random seed for reproducibility.

    Returns:
        An unfitted scikit-learn classifier.
    """
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"Unsupported estimator: {name}")


def expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand ``{"a": [1, 2], "b": [3]}`` into ``[{"a": 1, "b": 3}, {"a": 2, "b": 3}]``."""
    if not param_grid:
        return [{}]
    keys = sorted(param_grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(param_grid[k] for k in keys))
    ]


def select_model(
    X: pd.DataFrame, y: pd.Series, config: AppConfig
) -> tuple[CandidateResult, list[CandidateResult]]:
    """Cross-validate every candidate configuration as a nested MLflow run.

    Must be called inside an active parent run.

    Returns:
        The best candidate and the full list of results.
    """
    cv = StratifiedKFold(
        n_splits=config.training.cv_folds, shuffle=True, random_state=config.project.random_seed
    )
    metric = config.training.selection_metric
    results: list[CandidateResult] = []
    candidates: list[ModelCandidate] = config.training.candidates
    for candidate in candidates:
        for params in expand_grid(candidate.param_grid):
            pipeline = build_pipeline(
                build_estimator(candidate.estimator, params, config.project.random_seed)
            )
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=CV_SCORING, n_jobs=1)
            cv_mean = {m: float(np.mean(scores[f"test_{m}"])) for m in CV_SCORING}
            cv_std = {m: float(np.std(scores[f"test_{m}"])) for m in CV_SCORING}
            with mlflow.start_run(run_name=candidate.name, nested=True) as child:
                mlflow.log_params({"estimator": candidate.estimator, **params})
                mlflow.log_metrics({f"cv_{m}_mean": v for m, v in cv_mean.items()})
                mlflow.log_metrics({f"cv_{m}_std": v for m, v in cv_std.items()})
                mlflow.log_metric("cv_fit_time_mean", float(np.mean(scores["fit_time"])))
            logger.info(
                "CV %-24s %-45s %s=%.4f (+/- %.4f)",
                candidate.name,
                json.dumps(params, sort_keys=True),
                metric,
                cv_mean[metric],
                cv_std[metric],
            )
            results.append(
                CandidateResult(
                    candidate.name, candidate.estimator, params, cv_mean, cv_std, child.info.run_id
                )
            )
    best = max(results, key=lambda r: r.cv_mean[metric])
    return best, results


def _log_dataset(frame: pd.DataFrame, source: str, context: str) -> None:
    """Record dataset lineage (schema, digest, row count) on the active run."""
    typed = coerce_types(frame[[*feature_names(), TARGET_COLUMN]])
    typed[TARGET_COLUMN] = typed[TARGET_COLUMN].astype(float)
    try:
        with warnings.catch_warnings():
            # MLflow warns that a plain path "can be interpreted in multiple ways"; it
            # resolves to a local artifact source, which is what we want.
            warnings.filterwarnings("ignore", message=".*can be interpreted in multiple ways.*")
            dataset = from_pandas(typed, source=source, name=context, targets=TARGET_COLUMN)
        mlflow.log_input(dataset, context=context)
    except MlflowException:
        # Lineage metadata must never block training; the data digest is still in params.
        logger.warning("Could not record dataset lineage for source %r", source, exc_info=True)


def _pip_requirements() -> list[str]:
    """Pin the libraries needed to load the model artifact."""
    return [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"skops=={skops.__version__}",
    ]


def _log_reference_data(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: AppConfig,
    workdir: Path,
) -> None:
    """Log the data the monitor compares production traffic against."""
    reference = train_df[[*feature_names(), TARGET_COLUMN]]
    if len(reference) > config.training.reference_max_rows:
        reference = reference.sample(
            n=config.training.reference_max_rows, random_state=config.project.random_seed
        )
    X_eval, y_eval = split_features_target(eval_df)
    predictions = pd.DataFrame(
        {
            "churn_probability": pipeline.predict_proba(X_eval)[:, 1],
            TARGET_COLUMN: y_eval.to_numpy(),
        }
    )
    ref_dir = workdir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    reference.to_csv(ref_dir / "reference_data.csv", index=False)
    predictions.to_csv(ref_dir / "evaluation_predictions.csv", index=False)
    mlflow.log_artifacts(str(ref_dir), artifact_path="reference")


def _evaluate_champion(
    config: AppConfig, client: MlflowClient, eval_df: pd.DataFrame
) -> tuple[str | None, dict[str, float] | None]:
    """Score the current champion (if any) on the challenger's evaluation data."""
    champion = get_version_by_alias(
        client, config.registry.model_name, config.registry.champion_alias
    )
    if champion is None:
        return None, None
    loaded = load_model(
        config, model_uri=f"models:/{config.registry.model_name}/{champion.version}", client=client
    )
    X_eval, y_eval = split_features_target(eval_df)
    metrics = compute_metrics(
        y_eval, loaded.predict_proba(X_eval), config.training.decision_threshold
    )
    return str(champion.version), metrics


def train_and_register(
    config: AppConfig,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    promote: bool = True,
    data_source: str = "unknown",
) -> TrainingResult:
    """Select, fit, evaluate, log, register and (maybe) promote a model.

    Args:
        config: Application configuration.
        train_df: Labeled training rows.
        eval_df: Labeled evaluation rows (held out from training).
        run_name: MLflow run name.
        tags: Extra MLflow run tags (e.g. ``{"trigger": "drift"}``).
        promote: Run the promotion gate; when False the version is only tagged challenger.
        data_source: Human-readable provenance of the training data.

    Returns:
        The training result, including the promotion decision.
    """
    client = configure_mlflow(config)
    mlflow.set_experiment(config.training.experiment_name)
    name = config.registry.model_name
    threshold = config.training.decision_threshold
    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)

    with (
        mlflow.start_run(run_name=run_name, tags=tags) as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        workdir = Path(tmp)
        mlflow.log_params(
            {
                "data_source": data_source,
                "n_train": len(train_df),
                "n_eval": len(eval_df),
                "train_churn_rate": round(float(y_train.mean()), 4),
                "cv_folds": config.training.cv_folds,
                "selection_metric": config.training.selection_metric,
                "decision_threshold": threshold,
                "random_seed": config.project.random_seed,
            }
        )
        _log_dataset(train_df, data_source, "train")
        _log_dataset(eval_df, data_source, "evaluation")

        best, candidates = select_model(X_train, y_train, config)
        mlflow.log_params({"best_candidate": best.name, "best_estimator": best.estimator})
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metrics({f"cv_{k}_mean": v for k, v in best.cv_mean.items()})

        pipeline = build_pipeline(
            build_estimator(best.estimator, best.params, config.project.random_seed)
        )
        pipeline.fit(X_train, y_train)

        eval_dir = workdir / "evaluation"
        test_metrics = evaluate_model(
            pipeline, X_eval, y_eval, threshold, output_dir=eval_dir, with_importance=True
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if np.isfinite(v)})
        mlflow.log_artifacts(str(eval_dir), artifact_path="evaluation")
        _log_reference_data(pipeline, train_df, eval_df, config, workdir)

        example = X_train.dropna().head(5)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(example, pipeline.predict(example)),
            input_example=example,
            pip_requirements=_pip_requirements(),
            serialization_format="skops",
            skops_trusted_types=TRUSTED_TYPES,
        )
        version = register_model(
            model_info.model_uri,
            name,
            tags={
                "estimator": best.estimator,
                "data_source": data_source,
                "run_id": run.info.run_id,
            },
        )
        new_version = str(version.version)
        set_alias(client, name, config.registry.challenger_alias, new_version)

        champion_version, champion_metrics = _evaluate_champion(config, client, eval_df)
        if champion_metrics is not None:
            mlflow.log_metrics(
                {f"champion_{k}": v for k, v in champion_metrics.items() if np.isfinite(v)}
            )
        decision = decide_promotion(test_metrics, champion_metrics, config.registry.promotion)
        if not promote:
            decision = PromotionDecision(
                False,
                "promotion disabled for this run",
                decision.metric,
                decision.challenger_score,
                decision.champion_score,
            )
        record_decision(client, name, new_version, decision, champion_version)
        mlflow.set_tags(
            {
                "promotion.decision": "promoted" if decision.promote else "rejected",
                "promotion.reason": decision.reason,
            }
        )
        write_json(decision.as_tags(), workdir / "promotion_decision.json")
        mlflow.log_artifact(str(workdir / "promotion_decision.json"))

        if decision.promote:
            set_alias(client, name, config.registry.champion_alias, new_version)
            champion_version = new_version
        logger.info("Promotion decision for v%s: %s", new_version, decision.reason)

    return TrainingResult(
        run_id=run.info.run_id,
        model_name=name,
        model_version=new_version,
        best_candidate=best.name,
        best_params=best.params,
        test_metrics=test_metrics,
        decision=decision,
        champion_version=champion_version,
        candidates=candidates,
    )


def run_training(
    config: AppConfig,
    *,
    run_name: str | None = "initial-training",
    tags: dict[str, str] | None = None,
    promote: bool = True,
) -> TrainingResult:
    """Full pipeline from the configured data source to a registered model.

    Args:
        config: Application configuration.
        run_name: MLflow run name.
        tags: Extra MLflow run tags.
        promote: Whether the promotion gate may move the champion alias.

    Returns:
        The training result.
    """
    frame, snapshot = ingest(config)
    report = assert_valid(
        frame,
        require_target=True,
        min_rows=config.data.min_rows,
        max_missing_fraction=config.data.max_missing_fraction,
    )
    logger.info("Data contract passed: %d rows, %d warning(s)", report.n_rows, len(report.issues))
    train_df, eval_df = split_data(
        frame, test_size=config.data.test_size, random_state=config.project.random_seed
    )
    return train_and_register(
        config,
        train_df,
        eval_df,
        run_name=run_name,
        tags={"trigger": "manual", **(tags or {})},
        promote=promote,
        data_source=str(snapshot),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--run-name", default="initial-training", help="MLflow run name")
    parser.add_argument(
        "--skip-if-champion-exists",
        action="store_true",
        help="Exit successfully without training when a champion is already registered",
    )
    parser.add_argument("--no-promote", action="store_true", help="Register without promoting")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    if args.skip_if_champion_exists:
        client = configure_mlflow(config)
        champion = get_version_by_alias(
            client, config.registry.model_name, config.registry.champion_alias
        )
        if champion is not None:
            logger.info(
                "Champion already registered (%s v%s); skipping training",
                config.registry.model_name,
                champion.version,
            )
            return

    result = run_training(config, run_name=args.run_name, promote=not args.no_promote)
    print(json.dumps(result.summary(), indent=2, default=str))


if __name__ == "__main__":
    main()
