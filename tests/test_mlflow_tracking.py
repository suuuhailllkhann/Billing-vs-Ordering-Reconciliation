from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd

from pharmacy_reconciliation.research.mlflow_tracking import (
    BASELINE_EXPERIMENT,
    FINAL_EXPERIMENT,
    TUNED_EXPERIMENT,
    log_modeling_history,
)

OBSERVATIONS = Path("data/synthetic/longitudinal/refill_observations.csv")


def test_local_history_logs_nine_safe_deterministic_runs() -> None:
    Path("tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir="tmp") as directory:
        tracking = Path(directory) / "mlruns"
        ids = log_modeling_history(
            pd.read_csv(OBSERVATIONS), tracking_directory=tracking, log_final_artifact=False
        )
        assert {key: len(value) for key, value in ids.items()} == {
            "baseline": 4, "tuned": 4, "final": 1
        }
        client = mlflow.MlflowClient(tracking_uri=tracking.resolve().as_uri())
        experiments = {
            experiment.name: experiment for experiment in client.search_experiments()
        }
        assert {BASELINE_EXPERIMENT, TUNED_EXPERIMENT, FINAL_EXPERIMENT} <= set(experiments)
        final_runs = client.search_runs([experiments[FINAL_EXPERIMENT].experiment_id])
        assert len(final_runs) == 1
        final = final_runs[0]
        assert final.data.tags["model_status"] == "locked_final"
        assert final.data.tags["test_consumed"] == "true"
        assert final.data.params["C"] == "0.01"
        assert final.data.params["threshold"] == "0.5"
        assert final.data.params["fitting_policy"] == "Train only"
        assert "test_final_reporting_recall" in final.data.metrics

        forbidden = {"patient_id", "patient_name", "dob", "prescription_id", "ndc", "raw_csv"}
        for experiment in experiments.values():
            for run in client.search_runs([experiment.experiment_id]):
                assert forbidden.isdisjoint(run.data.params)
                assert forbidden.isdisjoint(run.data.tags)


def test_local_mlflow_store_is_gitignored() -> None:
    assert "mlruns/" in Path(".gitignore").read_text(encoding="utf-8").splitlines()
