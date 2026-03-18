"""``eda classify`` — train / evaluate classifiers on session data."""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from eda.data.loader import load_vault
from eda.data.schema import VALID_TARGETS
from eda.utils.plotting import plot_confusion_matrix, plot_feature_importance, save_fig

console = Console()

MODEL_MAP: dict[str, object] = {
    "random-forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "svm": LinearSVC(max_iter=2000, random_state=42),
    "logistic": LogisticRegression(max_iter=1000, n_jobs=-1, solver="lbfgs",
                                   multi_class="auto", random_state=42),
    "gradient-boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "naive-bayes": MultinomialNB(),
}

MODEL_CHOICES = click.Choice(list(MODEL_MAP.keys()))
TARGET_CHOICES = click.Choice(VALID_TARGETS)


@click.command("classify")
@click.option("--vault", "-v", required=True, type=click.Path(exists=True, file_okay=False),
              help="Obsidian vault root directory.")
@click.option("--target", "-t", default="track", type=TARGET_CHOICES, show_default=True,
              help="Column to predict.")
@click.option("--model", "-m", default="random-forest", type=MODEL_CHOICES,
              show_default=True, help="Classifier algorithm.")
@click.option("--output", "-o", default="results/classify",
              help="Output directory for model, report, and plots.")
@click.option("--cv-folds", default=5, show_default=True,
              help="Number of stratified cross-validation folds.")
@click.option("--max-tfidf", default=500, show_default=True,
              help="TF-IDF vocabulary size.")
@click.option("--no-workshops", is_flag=True, help="Exclude workshop sessions.")
@click.option("--no-save-model", is_flag=True, help="Skip saving the trained model.")
def classify_cmd(
    vault: str,
    target: str,
    model: str,
    output: str,
    cv_folds: int,
    max_tfidf: int,
    no_workshops: bool,
    no_save_model: bool,
) -> None:
    """Train a classifier to predict session track, level, or conference.

    Builds a TF-IDF → classifier pipeline and evaluates with
    stratified k-fold cross-validation.

    \b
    Output files
    ────────────
    <output>/model.joblib           — serialised sklearn Pipeline
    <output>/classification_report.json
    <output>/confusion_matrix.png
    <output>/feature_importance.png (tree-based models only)

    \b
    Examples
    ────────
    eda classify --vault ../.. --target track --model random-forest
    eda classify --vault . --target conference --model logistic --cv-folds 10
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load & filter data ------------------------------------------
    console.print(f"[bold blue]Loading sessions…[/]")
    df = load_vault(vault, include_workshops=not no_workshops)
    if df.empty:
        console.print("[red]No sessions found.[/]")
        raise SystemExit(1)

    if target not in df.columns:
        console.print(f"[red]Target column '{target}' not found. Available: {list(df.columns)}[/]")
        raise SystemExit(1)

    df = df[df[target].notna() & (df[target].astype(str).str.strip() != "")].copy()
    if len(df) < 20:
        console.print(f"[red]Too few labelled samples for target '{target}' ({len(df)}).[/]")
        raise SystemExit(1)

    console.print(f"[green]Samples: {len(df)} | Target classes: {df[target].nunique()}[/]")

    X_text = df["text"].fillna("").tolist()
    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str))

    # ---- build pipeline -----------------------------------------------
    estimator = MODEL_MAP[model]
    # MultinomialNB requires non-negative features — use absolute TF-IDF
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_tfidf,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=(model != "naive-bayes"),
        )),
        ("clf", estimator),
    ])

    # ---- cross-validation ---------------------------------------------
    console.print(f"[bold]Running {cv_folds}-fold CV with {model}…[/]")
    t0 = time.perf_counter()
    cv_results = cross_validate(
        pipeline,
        X_text,
        y,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring=["accuracy", "f1_weighted"],
        return_train_score=True,
        n_jobs=-1,
    )
    elapsed = time.perf_counter() - t0

    _print_cv_results(cv_results, elapsed)

    # ---- fit on full data + produce report ----------------------------
    pipeline.fit(X_text, y)
    y_pred = pipeline.predict(X_text)
    report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)

    with open(out / "classification_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    console.print(f"[green]Classification report → {out / 'classification_report.json'}[/]")

    # ---- confusion matrix plot ----------------------------------------
    cm = confusion_matrix(y, y_pred)
    fig = plot_confusion_matrix(cm, labels=list(le.classes_),
                                title=f"Confusion Matrix — {model} (target: {target})",
                                output=out / "confusion_matrix.png")
    console.print(f"[green]Confusion matrix → {out / 'confusion_matrix.png'}[/]")

    # ---- feature importance (tree models) ----------------------------
    clf = pipeline.named_steps["clf"]
    tfidf_vec = pipeline.named_steps["tfidf"]
    feature_names = tfidf_vec.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        plot_feature_importance(
            feature_names,
            clf.feature_importances_,
            top_n=30,
            title=f"Feature Importance — {model}",
            output=out / "feature_importance.png",
        )
        console.print(f"[green]Feature importance → {out / 'feature_importance.png'}[/]")
    elif hasattr(clf, "coef_"):
        # For linear models, show top coefficients for first class
        coef = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
        plot_feature_importance(
            feature_names, coef, top_n=30,
            title=f"Top Features (|coef|) — {model}",
            output=out / "feature_importance.png",
        )
        console.print(f"[green]Feature importance → {out / 'feature_importance.png'}[/]")

    # ---- save model ---------------------------------------------------
    if not no_save_model:
        model_path = out / "model.joblib"
        joblib.dump(pipeline, model_path)
        console.print(f"[bold green]Model saved → {model_path}[/]")

    # ---- save CV summary ----------------------------------------------
    cv_summary = {
        "model": model,
        "target": target,
        "cv_folds": cv_folds,
        "accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "accuracy_std": float(cv_results["test_accuracy"].std()),
        "f1_weighted_mean": float(cv_results["test_f1_weighted"].mean()),
        "f1_weighted_std": float(cv_results["test_f1_weighted"].std()),
        "classes": list(le.classes_),
    }
    with open(out / "cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(cv_summary, fh, indent=2)

    console.print(f"\n[bold green]Done. All outputs in {out}/[/]")


def _print_cv_results(cv_results: dict, elapsed: float) -> None:
    table = Table(title="Cross-Validation Results", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")

    for key in ("test_accuracy", "test_f1_weighted"):
        label = key.replace("test_", "").replace("_", " ").title()
        table.add_row(label,
                      f"{cv_results[key].mean():.4f}",
                      f"±{cv_results[key].std():.4f}")
    table.add_row("Elapsed (s)", f"{elapsed:.1f}", "")
    console.print(table)
