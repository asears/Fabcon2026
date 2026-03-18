"""I/O helpers supporting pandas optional I/O dependencies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, output: Path | str, fmt: str) -> Path:
    """Save *df* to *output* using the given *fmt*.

    Supported formats (and the pandas optional dependency they require):

    * ``csv``     — built-in
    * ``json``    — built-in
    * ``parquet`` — pyarrow >= 13
    * ``feather`` — pyarrow >= 13
    * ``excel``   — openpyxl >= 3.1
    * ``hdf5``    — tables >= 3.10
    * ``sqlite``  — SQLAlchemy >= 2.0

    Parameters
    ----------
    df:       DataFrame to save.
    output:   Destination path *without* extension.
    fmt:      One of the format strings above.

    Returns
    -------
    Path  The concrete file path that was written.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    match fmt:
        case "csv":
            dest = output.with_suffix(".csv")
            df.to_csv(dest, index=False)
        case "json":
            dest = output.with_suffix(".json")
            df.to_json(dest, orient="records", indent=2, date_format="iso")
        case "parquet":
            dest = output.with_suffix(".parquet")
            df.to_parquet(dest, index=False, engine="pyarrow")
        case "feather":
            dest = output.with_suffix(".feather")
            df.to_feather(dest)
        case "excel":
            dest = output.with_suffix(".xlsx")
            df.to_excel(dest, index=False, engine="openpyxl")
        case "hdf5":
            dest = output.with_suffix(".h5")
            df.to_hdf(dest, key="sessions", mode="w", complevel=5, complib="blosc")
        case "sqlite":
            import sqlalchemy  # noqa: F401 (optional dep check)

            dest = output.with_suffix(".db")
            from sqlalchemy import create_engine

            engine = create_engine(f"sqlite:///{dest}")
            df.to_sql("sessions", engine, if_exists="replace", index=False)
        case _:
            raise ValueError(
                f"Unknown format '{fmt}'. Choose: csv, json, parquet, feather, excel, hdf5, sqlite."
            )

    return dest


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Load a DataFrame from a file, inferring format from extension."""
    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path)
        case ".json":
            return pd.read_json(path, orient="records")
        case ".parquet":
            return pd.read_parquet(path, engine="pyarrow")
        case ".feather":
            return pd.read_feather(path)
        case ".xlsx" | ".xls":
            return pd.read_excel(path, engine="openpyxl")
        case ".h5" | ".hdf5":
            return pd.read_hdf(path, key="sessions")
        case ".db":
            from sqlalchemy import create_engine

            engine = create_engine(f"sqlite:///{path}")
            return pd.read_sql_table("sessions", engine)
        case _:
            raise ValueError(f"Cannot infer format from extension: {path.suffix}")
