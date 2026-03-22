"""``eda model-select`` — hyperparameter search over classifier candidates."""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from eda.data.loader import load_vault
from eda.data.schema import VALID_TARGETS
from eda.utils.plotting import plot_cv_comparison

console = Console()

TARGET_CHOICES = click.Choice(VALID_TARGETS)
SEARCH_CHOICES = click.Choice(["random", "grid"])

# ---- parameter grids ---------------------------------------------------

_PARAM_GRIDS: dict[str, dict] = {
    "random-forest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "tfidf__max_features": [300, 500, 1000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    },
    "logistic": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "saga"],
        "clf__max_iter": [500, 1000],
        "tfidf__max_features": [300, 500, 1000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    },
    "svm": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__max_iter": [1000, 2000],
        "tfidf__max_features": [300, 500, 1000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    },
    "gradient-boosting": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
        "clf__subsample": [0.8, 1.0],
        "tfidf__max_features": [300, 500],
    },
    "naive-bayes": {
        "clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
        "tfidf__max_features": [300, 500, 1000],
    },
}

_ESTIMATORS = {
    "random-forest": lambda: RandomForestClassifier(n_jobs=-1, random_state=42),
    "logistic": lambda: LogisticRegression(multi_class="auto", random_state=42),
    "svm": lambda: LinearSVC(random_state=42),
    "gradient-boosting": lambda: GradientBoostingClassifier(random_state=42),
    "naive-bayes": lambda: MultinomialNB(),
}


@click.command("model-select")
@click.option("--vault", "-v", required=True, type=click.Path(exists=True, file_okay=False),
              help="Obsidian vault root.")
@click.option("--target", "-t", default="track", type=TARGET_CHOICES, show_default=True,
              help="Column to predict.")
@click.option("--estimators", "-e", multiple=True,
              default=list(_ESTIMATORS.keys()), show_default=True,
              help="Which estimators to evaluate (repeat flag to specify multiple).")
@click.option("--search", default="random", type=SEARCH_CHOICES, show_default=True,
              help="Search strategy: random or grid.")
@click.option("--n-iter", default=20, show_default=True,
              help="Number of parameter settings sampled per estimator (random search only).")
@click.option("--cv-folds", default=5, show_default=True,
              help="Cross-validation folds.")
@click.option("--scoring", default="f1_weighted", show_default=True,
              help="sklearn scoring metric for search optimisation.")
@click.option("--output", "-o", default="results/model_select",
              help="Output directory.")
@click.option("--no-workshops", is_flag=True, help="Exclude workshop sessions.")
def model_select_cmd(
    vault: str,
    target: str,
    estimators: tuple[str, ...],
    search: str,
    n_iter: int,
    cv_folds: int,
    scoring: str,
    output: str,
    no_workshops: bool,
) -> None:
    """Compare classifiers with grid / randomised hyperparameter search.

    Runs RandomizedSearchCV or GridSearchCV for each estimator and
    produces a comparison chart of CV performance.

    \b
    Output files
    ────────────
    <output>/cv_comparison.png    — bar chart of CV scores
    <output>/cv_results.csv       — full results table
    <output>/best_params.json     — best params per estimator
    <output>/best_model.joblib    — top-performing estimator

    \b
    Examples
    ────────
    eda model-select --vault ../.. --target track
    eda model-select --vault . --target level --estimators random-forest --estimators logistic
    eda model-select --vault . --search grid --cv-folds 10 --scoring accuracy
    """
    import joblib  # lazy import

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # filter to valid estimator names
    valid_ests = [e for e in estimators if e in _ESTIMATORS]
    if not valid_ests:
        console.print(f"[red]No valid estimators. Choose from: {list(_ESTIMATORS)}[/]")
        raise SystemExit(1)

    console.print("[bold blue]Loading sessions…[/]")
    df = load_vault(vault, include_workshops=not no_workshops)
    if df.empty:
        console.print("[red]No sessions found.[/]")
        raise SystemExit(1)

    if target not in df.columns:
        console.print(f"[red]Target '{target}' not in columns.[/]")
        raise SystemExit(1)

    df = df[df[target].notna() & (df[target].astype(str).str.strip() != "")].copy()
    console.print(f"[green]Samples: {len(df)} | Target classes: {df[target].nunique()}[/]")

    X_text = df["text"].fillna("").tolist()
    y = _encode_target(df[target])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # ---- search -------------------------------------------------------
    results: list[dict] = []
    best_params_all: dict[str, dict] = {}
    best_score = -np.inf
    best_pipeline = None

    for name in valid_ests:
        console.print(f"\n[bold cyan]Searching {name}…[/]")
        pipeline = _build_pipeline(name)
        param_grid = _PARAM_GRIDS.get(name, {})

        if search == "random":
            searcher = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                refit=True,
            )
        else:
            from sklearn.model_selection import GridSearchCV

            searcher = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
            )

        searcher.fit(X_text, y)
        mean_score = searcher.best_score_
        std_score = float(searcher.cv_results_["std_test_score"][searcher.best_index_])
        console.print(
            f"  [green]Best {scoring}: {mean_score:.4f} ± {std_score:.4f}[/]"
            f"  params: {searcher.best_params_}"
        )

        best_params_all[name] = searcher.best_params_
        results.append({
            "model": name,
            f"cv_{scoring}_mean": mean_score,
            f"cv_{scoring}_std": std_score,
        })

        if mean_score > best_score:
            best_score = mean_score
            best_pipeline = searcher.best_estimator_

    # ---- summary table -----------------------------------------------
    _print_results(results, scoring)

    # ---- save outputs ------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "cv_results.csv", index=False)
    console.print(f"[green]CV results → {out / 'cv_results.csv'}[/]")

    with open(out / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best_params_all, fh, indent=2)
    console.print(f"[green]Best params → {out / 'best_params.json'}[/]")

    if best_pipeline is not None:
        joblib.dump(best_pipeline, out / "best_model.joblib")
        console.print(f"[green]Best model → {out / 'best_model.joblib'}[/]")

    # ---- comparison plot --------------------------------------------
    model_names = [r["model"] for r in results]
    means = [r[f"cv_{scoring}_mean"] for r in results]
    stds = [r[f"cv_{scoring}_std"] for r in results]
    plot_cv_comparison(model_names, means, stds, metric=scoring,
                       output=out / "cv_comparison.png")
    console.print(f"[green]Comparison chart → {out / 'cv_comparison.png'}[/]")

    console.print(f"\n[bold green]Done. Outputs in {out}/[/]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pipeline(name: str) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            sublinear_tf=(name != "naive-bayes"),
        )),
        ("clf", _ESTIMATORS[name]()),
    ])


def _encode_target(series: pd.Series) -> np.ndarray:
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    return le.fit_transform(series.astype(str))


def _print_results(results: list[dict], scoring: str) -> None:
    table = Table(title=f"Model Comparison — CV {scoring}", show_lines=False)
    table.add_column("Model", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")

    sorted_results = sorted(results, key=lambda r: r[f"cv_{scoring}_mean"], reverse=True)
    for r in sorted_results:
        table.add_row(
            r["model"],
            f"{r[f'cv_{scoring}_mean']:.4f}",
            f"±{r[f'cv_{scoring}_std']:.4f}",
        )
    console.print(table)
