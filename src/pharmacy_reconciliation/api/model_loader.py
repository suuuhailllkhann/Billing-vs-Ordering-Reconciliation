"""Load and validate the locked model from local MLflow SQLite tracking."""

from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd

from pharmacy_reconciliation.research.features import FEATURE_COLUMNS
from pharmacy_reconciliation.research.final_evaluation import LOCKED_THRESHOLD
from pharmacy_reconciliation.research.mlflow_tracking import FINAL_EXPERIMENT
from pharmacy_reconciliation.research.preparation import MODEL_FEATURE_COLUMNS

TRACKING_URI = "sqlite:///mlflow.db"
LOCKED_MODEL_NAME = "locked_pipeline"


def _contract_row() -> pd.DataFrame:
    row: dict[str, object] = {column: 0.0 for column in FEATURE_COLUMNS}
    row.update({
        "current_days_supply": 30,
        "average_previous_refill_interval_days": 30.0,
        "medication_prior_average_days_supply": 30.0,
        "std_previous_refill_interval_days": None,
    })
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def validate_locked_model(model: Any) -> None:
    """Fail clearly if the artifact does not match the locked serving contract."""
    if tuple(model.named_steps) != ("preprocessor", "scaler", "classifier"):
        raise RuntimeError("Locked model pipeline steps are incompatible.")
    transformed = model.named_steps["preprocessor"].transform(_contract_row())
    if tuple(transformed.columns) != MODEL_FEATURE_COLUMNS:
        raise RuntimeError("Locked model feature contract is incompatible.")
    classifier = model.named_steps["classifier"]
    if not (
        classifier.C == 0.01
        and classifier.solver == "saga"
        and classifier.max_iter == 5000
        and LOCKED_THRESHOLD == 0.50
    ):
        raise RuntimeError("Locked model configuration is incompatible.")


def load_locked_model(tracking_uri: str = TRACKING_URI) -> Any:
    """Discover the sole locked-final MLflow run and load its pipeline artifact."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(FINAL_EXPERIMENT)
    if experiment is None:
        raise RuntimeError("Locked-final MLflow experiment was not found.")
    runs = client.search_runs(
        [experiment.experiment_id], filter_string="tags.model_status = 'locked_final'"
    )
    if len(runs) != 1 or runs[0].data.tags.get("test_consumed") != "true":
        raise RuntimeError("Exactly one locked-final MLflow run is required.")
    models = [
        item for item in client.search_logged_models(experiment_ids=[experiment.experiment_id])
        if item.name == LOCKED_MODEL_NAME and item.source_run_id == runs[0].info.run_id
    ]
    if len(models) != 1:
        raise RuntimeError("Locked-final model artifact was not found.")
    model = mlflow_sklearn.load_model(models[0].model_uri)
    validate_locked_model(model)
    return model
