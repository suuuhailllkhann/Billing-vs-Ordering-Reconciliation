"""Local MLflow logging for the established Phase 2 modeling history."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd

from pharmacy_reconciliation.research.baselines import (
    BASELINE_RANDOM_SEED,
    BASELINE_THRESHOLD,
    compare_baselines,
    prepare_baseline_partitions,
)
from pharmacy_reconciliation.research.features import TARGET_COLUMN
from pharmacy_reconciliation.research.final_evaluation import evaluate_locked_test
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS, chronological_split
from pharmacy_reconciliation.research.threshold_analysis import tuned_full_feature_pipelines
from pharmacy_reconciliation.research.tuning import (
    TUNING_CV_SPLITS,
    TUNING_SCORING,
    TUNING_TRIALS,
    prepare_tuning_partitions,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
DEFAULT_TRACKING_DIRECTORY = REPOSITORY_ROOT / "mlruns"
BASELINE_EXPERIMENT = "pharmacy-reconciliation-baselines"
TUNED_EXPERIMENT = "pharmacy-reconciliation-tuned"
FINAL_EXPERIMENT = "pharmacy-reconciliation-locked-final"
SAFE_TAGS = {
    "dataset_type": "synthetic",
    "target": TARGET_COLUMN,
    "feature_set": "full_16",
}
TUNED_CV_SCORES = {
    "Logistic Regression": 0.7292, "Random Forest": 0.7502,
    "XGBoost": 0.7404, "LightGBM": 0.7398,
}


def _configure(tracking_directory: Path) -> None:
    tracking_directory.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_directory.resolve().as_uri())


def _log_metrics(prefix: str, metrics: Any) -> None:
    values = asdict(metrics)
    matrix = values.pop("confusion_matrix")
    mlflow.log_metrics({f"{prefix}_{key}": value for key, value in values.items()})
    mlflow.log_metrics({
        f"{prefix}_tn": matrix[0][0], f"{prefix}_fp": matrix[0][1],
        f"{prefix}_fn": matrix[1][0], f"{prefix}_tp": matrix[1][1],
    })


def _base_tags(stage: str, status: str, test_consumed: bool) -> dict[str, str]:
    return {
        **SAFE_TAGS, "run_stage": stage, "model_status": status,
        "test_consumed": str(test_consumed).lower(),
    }


def log_modeling_history(
    observations: pd.DataFrame,
    tracking_directory: Path = DEFAULT_TRACKING_DIRECTORY,
    log_final_artifact: bool = True,
) -> dict[str, list[str]]:
    """Log 4 baseline, 4 fixed tuned-history, and 1 locked final run locally."""
    _configure(tracking_directory)
    run_ids: dict[str, list[str]] = {"baseline": [], "tuned": [], "final": []}

    baseline_results = compare_baselines(prepare_baseline_partitions(observations))
    mlflow.set_experiment(BASELINE_EXPERIMENT)
    for result in baseline_results:
        with mlflow.start_run(run_name=result.model_name) as run:
            mlflow.set_tags(_base_tags("baseline", "candidate", False))
            mlflow.log_params({
                "model_name": result.model_name, "feature_count": 16,
                "feature_version": "full_16", "random_seed": BASELINE_RANDOM_SEED,
                "train_end_exclusive": "2026-02-01",
                "validation_end_inclusive": "2026-03-31",
                "preprocessing": "train_median_indicator; scaler_for_logistic",
                "threshold": BASELINE_THRESHOLD,
                **{f"model_{key}": str(value) for key, value in result.configuration.items()},
            })
            _log_metrics("train", result.train)
            _log_metrics("validation", result.validation)
            run_ids["baseline"].append(run.info.run_id)

    tuned_partitions = prepare_tuning_partitions(observations)
    mlflow.set_experiment(TUNED_EXPERIMENT)
    for model_name, model in tuned_full_feature_pipelines().items():
        model.fit(tuned_partitions.train.features, tuned_partitions.train.target)
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tags(_base_tags("tuned", "candidate", False))
            classifier = model.named_steps["classifier"]
            mlflow.log_params({
                "model_name": model_name, "feature_count": 16,
                "random_seed": BASELINE_RANDOM_SEED, "threshold": BASELINE_THRESHOLD,
                "tuning_method": "RandomizedSearchCV",
                "time_series_splits": TUNING_CV_SPLITS, "scoring": TUNING_SCORING,
                "trial_count": TUNING_TRIALS[model_name],
                **{f"model_{key}": str(value) for key, value in classifier.get_params(deep=False).items()},
            })
            mlflow.log_metric("best_cv_pr_auc", TUNED_CV_SCORES[model_name])
            from pharmacy_reconciliation.research.baselines import evaluate_classifier
            _log_metrics("train", evaluate_classifier(model, tuned_partitions.train))
            _log_metrics("validation", evaluate_classifier(model, tuned_partitions.validation))
            run_ids["tuned"].append(run.info.run_id)

    split = chronological_split(observations)
    final_model = tuned_full_feature_pipelines()["Logistic Regression"]
    final_model.fit(split.train.loc[:, list(MODEL_FEATURE_COLUMNS[:-1])], split.train[TARGET_COLUMN])
    final_metrics = evaluate_locked_test(observations)
    mlflow.set_experiment(FINAL_EXPERIMENT)
    with mlflow.start_run(run_name="locked-logistic-regression") as run:
        mlflow.set_tags(_base_tags("final", "locked_final", True))
        mlflow.log_params({
            "model_name": "Logistic Regression", "feature_count": 16,
            "C": 0.01, "penalty": "l2", "solver": "saga", "max_iter": 5000,
            "random_seed": BASELINE_RANDOM_SEED, "threshold": 0.50,
            "fitting_policy": "Train only", "preprocessing": "train_median_indicator_and_StandardScaler",
        })
        for prefix, metrics in (
            ("validation", final_metrics.validation), ("test_final_reporting", final_metrics.overall),
            ("test_seen", final_metrics.seen_patients), ("test_unseen", final_metrics.unseen_patients),
        ):
            mlflow.log_metrics({f"{prefix}_{key}": float(value) for key, value in metrics.items()})
        mlflow.log_dict({"feature_order": list(MODEL_FEATURE_COLUMNS)}, "model_contract.json")
        if log_final_artifact:
            mlflow_sklearn.log_model(
                final_model,
                name="locked_pipeline",
                skops_trusted_types=[
                    "pharmacy_reconciliation.research.tuning.FoldLocalRefillPreprocessor"
                ],
            )
        run_ids["final"].append(run.info.run_id)
    return run_ids
