"""Environment-only database configuration without credential exposure."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

APPLICATION_DATABASE_USER = "pharmacy_app"


@dataclass(frozen=True)
class DatabaseSettings:
    database_url: URL


def load_database_settings(env_file: Path | str = ".env") -> DatabaseSettings:
    load_dotenv(dotenv_path=env_file, override=False)
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("Persistence is enabled but DATABASE_URL is not configured.")
    try:
        parts = urlsplit(database_url)
        url = URL.create(
            drivername=parts.scheme,
            username=unquote(parts.username) if parts.username else None,
            password=unquote(parts.password) if parts.password else None,
            host=parts.hostname,
            port=parts.port,
            database=unquote(parts.path.lstrip("/")),
            query=dict(parse_qsl(parts.query)),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("DATABASE_URL is invalid.") from error
    if url.drivername not in {"postgresql", "postgresql+psycopg"}:
        raise RuntimeError("DATABASE_URL must use PostgreSQL with psycopg.")
    if url.username != APPLICATION_DATABASE_USER:
        raise RuntimeError("DATABASE_URL must use the configured application database user.")
    return DatabaseSettings(database_url=url)


def create_database_engine(settings: DatabaseSettings) -> Engine:
    return create_engine(settings.database_url, pool_pre_ping=True)


def verify_database(engine: Engine) -> None:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as error:
        raise RuntimeError("Persistence database is unavailable or invalid.") from error


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
