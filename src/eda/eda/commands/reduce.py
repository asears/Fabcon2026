"""``eda reduce`` — project sessions into 2-D / 3-D via PCA / t-SNE / UMAP."""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer

from eda.data.loader import load_vault
from eda.utils.plotting import plot_scatter_2d, plot_scatter_3d

console = Console()

METHOD_CHOICES = click.Choice(["pca", "tsne", "umap"])
COLOR_CHOICES = click.Choice(["track", "level", "conference", "day", "session_type", "cluster"])


@click.command("reduce")
@click.option("--vault", "-v", required=True, type=click.Path(exists=True, file_okay=False),
              help="Obsidian vault root.")
@click.option("--method", "-m", default="pca", type=METHOD_CHOICES, show_default=True,
              help="Dimensionality reduction algorithm.")
@click.option("--components", "-n", default=2, show_default=True,
              help="Number of output dimensions (2 or 3).")
@click.option("--color-by", default="track", type=COLOR_CHOICES, show_default=True,
              help="Column to use for scatter plot colour coding.")
@click.option("--max-tfidf", default=500, show_default=True,
              help="TF-IDF vocabulary size.")
@click.option("--tsne-perplexity", default=30, show_default=True,
              help="t-SNE perplexity (typical range 5–50).")
@click.option("--tsne-iterations", default=1000, show_default=True,
              help="t-SNE max iterations.")
@click.option("--umap-neighbors", default=15, show_default=True,
              help="UMAP n_neighbors (requires umap-learn extra).")
@click.option("--pca-whiten", is_flag=True,
              help="Apply PCA whitening.")
@click.option("--output", "-o", default="results/reduce",
              help="Output directory.")
@click.option("--no-workshops", is_flag=True, help="Exclude workshop sessions.")
def reduce_cmd(
    vault: str,
    method: str,
    components: int,
    color_by: str,
    max_tfidf: int,
    tsne_perplexity: int,
    tsne_iterations: int,
    umap_neighbors: int,
    pca_whiten: bool,
    output: str,
    no_workshops: bool,
) -> None:
    """Project high-dimensional session features into 2-D / 3-D space.

    \b
    Algorithms
    ──────────
    pca   — Principal Component Analysis (fast, linear)
    tsne  — t-Distributed Stochastic Neighbour Embedding (non-linear, slow)
    umap  — Uniform Manifold Approximation (requires pip install eda[umap])

    \b
    Output files
    ────────────
    <output>/embeddings.csv   — session title + 2-D/3-D coordinates
    <output>/scatter.png      — colour-coded scatter plot

    \b
    Examples
    ────────
    eda reduce --vault ../.. --method pca --color-by track
    eda reduce --vault . --method tsne --components 2 --color-by level
    eda reduce --vault . --method umap --color-by conference
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    if components not in (2, 3):
        console.print("[red]--components must be 2 or 3.[/]")
        raise SystemExit(1)

    console.print("[bold blue]Loading sessions…[/]")
    df = load_vault(vault, include_workshops=not no_workshops)
    if df.empty:
        console.print("[red]No sessions found.[/]")
        raise SystemExit(1)

    console.print(f"[green]Loaded {len(df)} sessions[/]")

    # ---- TF-IDF --------------------------------------------------------
    tfidf = TfidfVectorizer(
        max_features=max_tfidf,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_sparse = tfidf.fit_transform(df["text"].fillna("").tolist())
    X = X_sparse.toarray()

    # ---- dimensionality reduction ------------------------------------
    console.print(f"[cyan]Running {method.upper()} → {components} components…[/]")

    embedding = _run_reduction(method, X, components,
                               tsne_perplexity, tsne_iterations,
                               umap_neighbors, pca_whiten)

    # ---- colour labels -----------------------------------------------
    if color_by in df.columns:
        color_labels = df[color_by].fillna("unknown").astype(str).tolist()
    else:
        console.print(f"[yellow]Column '{color_by}' not found, using index.[/]")
        color_labels = [str(i) for i in range(len(df))]

    # ---- save embeddings CSV -----------------------------------------
    embed_df = pd.DataFrame()
    if "title" in df.columns:
        embed_df["title"] = df["title"].values
    if "track" in df.columns:
        embed_df["track"] = df["track"].values
    embed_df["c1"] = embedding[:, 0]
    embed_df["c2"] = embedding[:, 1]
    if components == 3:
        embed_df["c3"] = embedding[:, 2]
    embed_df[color_by] = color_labels
    embed_df.to_csv(out / "embeddings.csv", index=False)
    console.print(f"[green]Embeddings → {out / 'embeddings.csv'}[/]")

    # ---- scatter plot ------------------------------------------------
    title = f"{method.upper()} Projection — coloured by {color_by}"
    if components == 2:
        plot_scatter_2d(
            embedding[:, 0], embedding[:, 1],
            color_labels,
            title=title,
            xlabel=f"{method.upper()} 1",
            ylabel=f"{method.upper()} 2",
            output=out / "scatter.png",
        )
    else:
        plot_scatter_3d(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            color_labels,
            title=title,
            output=out / "scatter.png",
        )
    console.print(f"[green]Scatter → {out / 'scatter.png'}[/]")
    console.print(f"\n[bold green]Done. Outputs in {out}/[/]")


# ---------------------------------------------------------------------------
# Algorithm helpers
# ---------------------------------------------------------------------------


def _run_reduction(
    method: str,
    X: np.ndarray,
    n_components: int,
    tsne_perplexity: int,
    tsne_iterations: int,
    umap_neighbors: int,
    pca_whiten: bool,
) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        reduced = PCA(n_components=n_components, whiten=pca_whiten, random_state=42).fit_transform(X)
        console.print(f"[green]PCA done. Shape: {reduced.shape}[/]")
        return reduced

    if method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE works best on PCA-pre-reduced data when features > 50
        if X.shape[1] > 50:
            from sklearn.decomposition import PCA as _PCA
            X = _PCA(n_components=50, random_state=42).fit_transform(X)
        reduced = TSNE(
            n_components=n_components,
            perplexity=tsne_perplexity,
            max_iter=tsne_iterations,
            random_state=42,
            n_jobs=-1,
        ).fit_transform(X)
        console.print(f"[green]t-SNE done. Shape: {reduced.shape}[/]")
        return reduced

    if method == "umap":
        try:
            import umap  # noqa: F401
        except ImportError:
            console.print(
                "[red]UMAP not installed. Run: pip install 'eda[umap]' or uv add umap-learn[/]"
            )
            raise SystemExit(1)
        import umap as umap_lib

        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=umap_neighbors,
            random_state=42,
        )
        reduced = reducer.fit_transform(X)
        console.print(f"[green]UMAP done. Shape: {reduced.shape}[/]")
        return reduced

    raise ValueError(f"Unknown method: {method}")
