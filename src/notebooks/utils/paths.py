from pathlib import Path


def resolve_workspace(start: Path | None = None) -> Path:
    """Resolve the repository root from common notebook working directories."""
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for path in candidates:
        if (path / "Processed").exists() and ((path / "src").exists() or (path / "Sessions").exists()):
            return path
    for path in candidates:
        if (path / "src" / "eda").exists():
            return path
    return current


def get_db_paths(workspace: Path) -> tuple[Path, Path]:
    """Return (input_db, output_db) paths used throughout ML notebooks."""
    input_db = workspace / "Processed" / "sessions_in_preprocessed.duckdb"
    output_db = workspace / "Processed" / "sessions_ml_outputs.duckdb"
    return input_db, output_db
