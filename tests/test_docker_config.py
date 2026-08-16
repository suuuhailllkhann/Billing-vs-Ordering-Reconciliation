from pathlib import Path

import pytest

from pharmacy_reconciliation.api.model_loader import (
    DEFAULT_TRACKING_URI,
    FINAL_EXPERIMENT,
    resolve_model_source,
    resolve_tracking_uri,
)


def test_mlflow_runtime_paths_are_environment_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_ROOT", raising=False)
    assert resolve_tracking_uri() == DEFAULT_TRACKING_URI
    assert resolve_model_source("model-id", "models:/model-id") == "models:/model-id"

    root = Path(".docker-test-artifacts")
    mounted_model = root / FINAL_EXPERIMENT / "models" / "model-id" / "artifacts"
    mounted_model.mkdir(parents=True)
    try:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////app/runtime/mlflow.db")
        monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(root))
        assert resolve_tracking_uri() == "sqlite:////app/runtime/mlflow.db"
        assert Path(resolve_model_source("model-id", "unused")) == mounted_model
    finally:
        for directory in [mounted_model, *mounted_model.parents[:4]]:
            if directory.exists() and directory != Path("."):
                directory.rmdir()


def test_dockerfile_uses_locked_uv_environment_and_safe_runtime() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "uv sync --locked --no-dev --extra api --extra ml" in dockerfile
    assert '"--host", "0.0.0.0", "--port", "8000"' in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "DATABASE_URL" not in dockerfile
    assert ".env" not in dockerfile


def test_dockerignore_excludes_secrets_and_runtime_mounts() -> None:
    ignored = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".env", ".env.*", "mlflow.db", "mlflow_artifacts/", ".git/", ".venv/"} <= ignored
