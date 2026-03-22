from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

import duckdb


@contextmanager
def open_duckdb(
    db_path: Path | str,
    read_only: bool = False,
    retries: int = 6,
    retry_wait_seconds: float = 0.5,
):
    """Open a DuckDB connection with retry and always close it."""
    path_str = str(db_path)
    last_error = None
    for attempt in range(retries):
        try:
            con = duckdb.connect(path_str, read_only=read_only)
            break
        except duckdb.IOException as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(retry_wait_seconds)
    else:
        raise last_error if last_error else RuntimeError("Failed to open DuckDB connection.")

    try:
        yield con
    finally:
        con.close()


def query_df(
    db_path: Path | str,
    sql: str,
    params: list | tuple | None = None,
    read_only: bool = True,
):
    """Execute a query and return a DataFrame with auto-managed connection lifetime."""
    with open_duckdb(db_path, read_only=read_only) as con:
        if params is None:
            return con.execute(sql).df()
        return con.execute(sql, params).df()


def init_ml_artifacts_table(output_db_path: Path | str) -> None:
    """Create the shared artifact storage table if it does not already exist."""
    with open_duckdb(output_db_path, read_only=False) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_artifacts (
              artifact_id VARCHAR PRIMARY KEY,
              notebook VARCHAR,
              artifact_type VARCHAR,
              model_name VARCHAR,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              metrics_json VARCHAR,
              artifact_blob BLOB
            )
            """
        )


def table_exists(db_path: Path | str, table_name: str) -> bool:
    """Check whether a table exists in the target DuckDB database."""
    with open_duckdb(db_path, read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    return table_name in tables
