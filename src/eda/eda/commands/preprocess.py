"""``eda preprocess`` — build ML-ready feature matrices from vault sessions."""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from eda.data.loader import load_vault
from eda.utils.io import save_dataframe

console = Console()

FORMAT_CHOICES = click.Choice(["csv", "json", "parquet", "feather", "excel", "hdf5", "sqlite"])


@click.command("preprocess")
@click.option("--vault", "-v", required=True, type=click.Path(exists=True, file_okay=False),
              help="Obsidian vault root directory.")
@click.option("--output", "-o", default="data/sessions",
              help="Output base path (extension added automatically).")
@click.option("--format", "-f", "fmt", default="parquet", type=FORMAT_CHOICES,
              help="Output format for the processed DataFrame.")
@click.option("--max-tfidf", default=500, show_default=True,
              help="Maximum TF-IDF vocabulary size.")
@click.option("--ngram-min", default=1, show_default=True, help="Minimum n-gram size.")
@click.option("--ngram-max", default=2, show_default=True, help="Maximum n-gram size.")
@click.option("--no-workshops", is_flag=True, help="Exclude workshop sessions.")
@click.option("--no-tfidf", is_flag=True, help="Skip TF-IDF computation.")
@click.option("--stats/--no-stats", default=True, help="Print dataset statistics.")
def preprocess_cmd(
    vault: str,
    output: str,
    fmt: str,
    max_tfidf: int,
    ngram_min: int,
    ngram_max: int,
    no_workshops: bool,
    no_tfidf: bool,
    stats: bool,
) -> None:
    """Load vault sessions and produce ML-ready feature matrices.

    \b
    Output files
    ────────────
    <output>.<ext>          — encoded DataFrame (all sessions as rows)
    <output>_tfidf.npz      — TF-IDF sparse matrix (scipy format)
    <output>_tfidf_vocab.csv— feature index → term mapping

    \b
    Pandas I/O optional dependencies used
    ──────────────────────────────────────
    parquet / feather → pyarrow
    excel             → openpyxl
    hdf5              → tables (PyTables)
    sqlite            → SQLAlchemy

    \b
    Examples
    ────────
    eda preprocess --vault ../.. --format parquet
    eda preprocess --vault . --output data/sessions --format csv --max-tfidf 1000
    """
    console.print(f"[bold blue]Loading sessions from:[/] {vault}")

    df = load_vault(vault, include_workshops=not no_workshops)
    if df.empty:
        console.print("[red]No sessions found.[/]")
        raise SystemExit(1)

    console.print(f"[green]Loaded {len(df)} sessions[/]")

    if stats:
        _print_stats(df)

    # ---- encode categorical columns ------------------------------------
    cat_cols = ["day", "conference", "session_type", "level_name", "status", "track"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        if mask.sum() > 0:
            encoded = pd.Series(np.full(len(df), -1, dtype=int), index=df.index)
            encoded[mask] = le.fit_transform(df.loc[mask, col].astype(str))
            df[f"{col}_encoded"] = encoded

    # ---- TF-IDF --------------------------------------------------------
    if not no_tfidf and "text" in df.columns:
        tfidf = TfidfVectorizer(
            max_features=max_tfidf,
            stop_words="english",
            ngram_range=(ngram_min, ngram_max),
            min_df=2,
            sublinear_tf=True,
        )
        docs = df["text"].fillna("").tolist()
        X_text: sp.csr_matrix = tfidf.fit_transform(docs)

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)

        npz_path = out.parent / (out.name + "_tfidf.npz")
        sp.save_npz(str(npz_path), X_text)
        console.print(f"[green]TF-IDF matrix {X_text.shape} → {npz_path}[/]")

        vocab_path = out.parent / (out.name + "_tfidf_vocab.csv")
        pd.DataFrame({
            "index": range(len(tfidf.get_feature_names_out())),
            "term": tfidf.get_feature_names_out(),
        }).to_csv(vocab_path, index=False)
        console.print(f"[green]Vocabulary → {vocab_path}[/]")

    # ---- save DataFrame ------------------------------------------------
    dest = save_dataframe(df, output, fmt)
    console.print(f"[bold green]Saved → {dest}[/]")


# ---------------------------------------------------------------------------
# Stats table helpers
# ---------------------------------------------------------------------------


def _print_stats(df: pd.DataFrame) -> None:
    table = Table(title="Dataset Overview", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Total sessions", str(len(df)))

    if "conference" in df.columns:
        for conf, cnt in df["conference"].value_counts().items():
            table.add_row(f"  {conf}", str(cnt))

    if "day" in df.columns:
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            cnt = (df["day"] == day).sum()
            if cnt:
                table.add_row(f"  {day}", str(cnt))

    if "track" in df.columns:
        table.add_row("Unique tracks", str(df["track"].nunique()))

    if "level" in df.columns:
        for lvl in [100, 200, 300, 400]:
            cnt = (df["level"] == lvl).sum()
            if cnt:
                table.add_row(f"  Level {lvl}", str(cnt))

    if "status" in df.columns:
        for status, cnt in df["status"].value_counts().items():
            table.add_row(f"  Status: {status}", str(cnt))

    console.print(table)
