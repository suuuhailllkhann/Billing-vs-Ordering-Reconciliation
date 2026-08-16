from pathlib import Path


def test_compose_uses_internal_database_network_and_health_dependency() -> None:
    compose = Path("compose.yaml").read_text(encoding="utf-8")
    assert "image: postgres:18" in compose
    assert "POSTGRES_DB: pharmacy_reconciliation" in compose
    assert "POSTGRES_USER: pharmacy_app" in compose
    assert "${POSTGRES_PASSWORD:?" in compose
    assert "@db:5432/pharmacy_reconciliation" in compose
    assert "pg_isready" in compose
    assert "condition: service_healthy" in compose
    assert "host.docker.internal" not in compose
    assert "localhost" not in compose


def test_compose_persists_postgres_18_and_mounts_mlflow_read_only() -> None:
    compose = Path("compose.yaml").read_text(encoding="utf-8")
    assert "pharmacy_postgres_data:/var/lib/postgresql" in compose
    assert "name: pharmacy_postgres_data" in compose
    assert "target: /app/runtime/mlflow.db" in compose
    assert "target: /app/runtime/mlflow_artifacts" in compose
    assert compose.count("read_only: true") == 2


def test_api_image_contains_explicit_migration_assets() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    ignored = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())
    assert "COPY alembic.ini ./" in dockerfile
    assert "COPY migrations ./migrations" in dockerfile
    assert "alembic.ini" not in ignored
    assert "migrations/" not in ignored


def test_compose_secret_file_is_ignored_but_template_is_tracked() -> None:
    gitignore = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())
    template = Path(".env.compose.example").read_text(encoding="utf-8")
    assert ".env.*" in gitignore
    assert "!.env.compose.example" in gitignore
    assert "replace_with_url_safe_local_secret" in template
