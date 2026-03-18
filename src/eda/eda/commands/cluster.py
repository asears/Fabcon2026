"""``eda cluster`` — unsupervised grouping of sessions."""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

from eda.data.loader import load_vault
from eda.utils.plotting import plot_elbow, plot_scatter_2d, plot_silhouette, save_fig

console = Console()

ALGO_CHOICES = click.Choice(["kmeans", "dbscan", "agglomerative"])


@click.command("cluster")
@click.option("--vault", "-v", required=True, type=click.Path(exists=True, file_okay=False),
              help="Obsidian vault root.")
@click.option("--algorithm", "-a", default="kmeans", type=ALGO_CHOICES, show_default=True,
              help="Clustering algorithm.")
@click.option("--n-clusters", "-k", default=0, show_default=True,
              help="Number of clusters. 0 = auto (uses number of unique tracks).")
@click.option("--k-min", default=2, show_default=True,
              help="Minimum k for elbow / silhouette sweep (kmeans only).")
@click.option("--k-max", default=25, show_default=True,
              help="Maximum k for elbow / silhouette sweep (kmeans only).")
@click.option("--dbscan-eps", default=0.5, show_default=True,
              help="DBSCAN epsilon neighbourhood radius.")
@click.option("--dbscan-min-samples", default=3, show_default=True,
              help="DBSCAN minimum samples per core point.")
@click.option("--max-tfidf", default=500, show_default=True,
              help="TF-IDF vocabulary size.")
@click.option("--output", "-o", default="results/cluster",
              help="Output directory.")
@click.option("--no-workshops", is_flag=True, help="Exclude workshop sessions.")
@click.option("--write-back", is_flag=True,
              help="Write ml_cluster property back to vault markdown files.")
def cluster_cmd(
    vault: str,
    algorithm: str,
    n_clusters: int,
    k_min: int,
    k_max: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    max_tfidf: int,
    output: str,
    no_workshops: bool,
    write_back: bool,
) -> None:
    """Discover natural groupings in sessions using unsupervised clustering.

    For KMeans, runs an elbow + silhouette sweep to suggest the best k
    unless ``--n-clusters`` is explicitly set.

    \b
    Output files
    ────────────
    <output>/cluster_labels.csv   — session title + assigned cluster
    <output>/elbow.png            — inertia curve (KMeans)
    <output>/silhouette.png       — silhouette scores vs k (KMeans)
    <output>/scatter.png          — 2-D PCA projection coloured by cluster

    \b
    Examples
    ────────
    eda cluster --vault ../.. --algorithm kmeans --n-clusters 10
    eda cluster --vault . --algorithm dbscan --dbscan-eps 0.3
    eda cluster --vault . --algorithm agglomerative -k 15 --write-back
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Loading sessions…[/]")
    df = load_vault(vault, include_workshops=not no_workshops)
    if df.empty:
        console.print("[red]No sessions found.[/]")
        raise SystemExit(1)

    console.print(f"[green]Loaded {len(df)} sessions[/]")

    # ---- TF-IDF features (L2-normalised for cosine distance) ----------
    tfidf = TfidfVectorizer(
        max_features=max_tfidf,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(df["text"].fillna("").tolist())
    X_dense = normalize(X).toarray()

    # ---- determine k --------------------------------------------------
    if n_clusters == 0 and "track" in df.columns:
        n_clusters = max(int(df["track"].nunique()), 5)
        console.print(f"[cyan]Auto-selected k={n_clusters} (from unique tracks)[/]")
    elif n_clusters == 0:
        n_clusters = 10

    # ---- algorithm-specific logic ------------------------------------
    if algorithm == "kmeans":
        labels = _run_kmeans(X_dense, n_clusters, k_min, k_max, out)
    elif algorithm == "dbscan":
        labels = _run_dbscan(X_dense, dbscan_eps, dbscan_min_samples)
    else:  # agglomerative
        labels = _run_agglomerative(X_dense, n_clusters, out)

    df["ml_cluster"] = labels

    # ---- quality metrics ---------------------------------------------
    _print_cluster_stats(df, labels)

    # ---- 2-D scatter via PCA -----------------------------------------
    from sklearn.decomposition import PCA  # lazy import

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_dense)

    plot_scatter_2d(
        X_2d[:, 0], X_2d[:, 1],
        labels=[f"C{l}" for l in labels],
        title=f"{algorithm.title()} Clusters (PCA 2-D)",
        output=out / "scatter.png",
    )
    console.print(f"[green]Scatter plot → {out / 'scatter.png'}[/]")

    # ---- save cluster labels -----------------------------------------
    result = df[["file", "file_path"] +
                (["title"] if "title" in df.columns else []) +
                (["track"] if "track" in df.columns else []) +
                ["ml_cluster"]].copy()
    result.to_csv(out / "cluster_labels.csv", index=False)
    console.print(f"[green]Cluster labels → {out / 'cluster_labels.csv'}[/]")

    # ---- optional write-back -----------------------------------------
    if write_back:
        _write_back_clusters(df)
        console.print("[bold green]ml_cluster property written back to vault files.[/]")

    console.print(f"\n[bold green]Done. Outputs in {out}/[/]")


# ---------------------------------------------------------------------------
# Algorithm helpers
# ---------------------------------------------------------------------------


def _run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    k_min: int,
    k_max: int,
    out: Path,
) -> np.ndarray:
    """Fit KMeans and produce elbow + silhouette plots."""
    k_range = range(k_min, min(k_max + 1, len(X)))
    inertias, sil_scores = [], []

    console.print(f"[cyan]Sweeping k={k_min}…{k_max} for elbow/silhouette…[/]")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, lbl, sample_size=min(1000, len(X))))

    plot_elbow(list(k_range), inertias, output=out / "elbow.png")
    plot_silhouette(list(k_range), sil_scores, output=out / "silhouette.png")
    console.print(f"[green]Elbow → {out / 'elbow.png'} | Silhouette → {out / 'silhouette.png'}[/]")

    # Fit with the requested (or auto-derived) k
    best_k = n_clusters
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = km_final.fit_predict(X)
    console.print(f"[green]KMeans k={best_k} | inertia={km_final.inertia_:.1f}[/]")
    return labels


def _run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    console.print(f"[green]DBSCAN: {n_clusters} clusters, {n_noise} noise points[/]")
    return labels


def _run_agglomerative(X: np.ndarray, n_clusters: int, out: Path) -> np.ndarray:
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(X)
    console.print(f"[green]Agglomerative: {n_clusters} clusters[/]")
    return labels


def _print_cluster_stats(df: pd.DataFrame, labels: np.ndarray) -> None:
    table = Table(title="Cluster Size Distribution", show_lines=False)
    table.add_column("Cluster", style="cyan")
    table.add_column("Sessions", style="green")
    table.add_column("Top Track", style="yellow")

    series = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in series.items():
        mask = labels == cluster_id
        top_track = ""
        if "track" in df.columns:
            top_track = df.loc[mask, "track"].value_counts().index[0] if mask.sum() else ""
        table.add_row(
            "Noise" if cluster_id == -1 else f"C{cluster_id}",
            str(count),
            str(top_track),
        )
    console.print(table)


def _write_back_clusters(df: pd.DataFrame) -> None:
    import frontmatter as fm

    for _, row in df.iterrows():
        path = Path(row["file_path"])
        if not path.exists():
            continue
        try:
            post = fm.load(str(path))
            post.metadata["ml_cluster"] = int(row["ml_cluster"])
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(fm.dumps(post))
        except Exception:
            continue
