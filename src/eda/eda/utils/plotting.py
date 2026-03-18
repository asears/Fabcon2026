"""Shared matplotlib helpers for CLI output plots."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig: plt.Figure, path: Path | str, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str = "Confusion Matrix",
    output: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    if output:
        save_fig(fig, output)
    return fig


def plot_scatter_2d(
    x: np.ndarray,
    y: np.ndarray,
    labels: Sequence,
    title: str = "2-D Projection",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    output: Path | str | None = None,
) -> plt.Figure:
    unique_labels = sorted(set(str(l) for l in labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    color_map = {l: cmap(i) for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for lbl in unique_labels:
        mask = np.array([str(l) == lbl for l in labels])
        ax.scatter(x[mask], y[mask], c=[color_map[lbl]], label=lbl, alpha=0.7, s=30)

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best", fontsize=7, ncol=max(1, len(unique_labels) // 20))
    fig.tight_layout()
    if output:
        save_fig(fig, output)
    return fig


def plot_scatter_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: Sequence,
    title: str = "3-D Projection",
    output: Path | str | None = None,
) -> plt.Figure:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    unique_labels = sorted(set(str(l) for l in labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    color_map = {l: cmap(i) for i, l in enumerate(unique_labels)}

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    for lbl in unique_labels:
        mask = np.array([str(l) == lbl for l in labels])
        ax.scatter(x[mask], y[mask], z[mask], c=[color_map[lbl]], label=lbl, alpha=0.7, s=20)
    ax.set(title=title, xlabel="C1", ylabel="C2", zlabel="C3")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    if output:
        save_fig(fig, output)
    return fig


def plot_elbow(
    k_values: Sequence[int],
    inertias: Sequence[float],
    output: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_values), list(inertias), "bo-")
    ax.set(xlabel="Number of Clusters (k)", ylabel="Inertia", title="KMeans Elbow Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output:
        save_fig(fig, output)
    return fig


def plot_silhouette(
    k_values: Sequence[int],
    scores: Sequence[float],
    output: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_values), list(scores), "go-")
    best_k = k_values[int(np.argmax(scores))]
    ax.axvline(best_k, color="r", linestyle="--", label=f"Best k={best_k}")
    ax.set(xlabel="Number of Clusters (k)", ylabel="Silhouette Score", title="Silhouette Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output:
        save_fig(fig, output)
    return fig


def plot_cv_comparison(
    model_names: Sequence[str],
    mean_scores: Sequence[float],
    std_scores: Sequence[float],
    metric: str = "accuracy",
    output: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(model_names))
    ax.barh(list(y_pos), list(mean_scores), xerr=list(std_scores),
            align="center", alpha=0.8, color="steelblue", capsize=4)
    ax.set(
        yticks=list(y_pos),
        yticklabels=list(model_names),
        xlabel=f"CV {metric}",
        title=f"Model Comparison — Cross-Validated {metric.title()}",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if output:
        save_fig(fig, output)
    return fig


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    top_n: int = 30,
    title: str = "Feature Importance",
    output: Path | str | None = None,
) -> plt.Figure:
    idx = np.argsort(importances)[-top_n:]
    features = [feature_names[i] for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(features, values, color="steelblue")
    ax.set(xlabel="Importance", title=title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if output:
        save_fig(fig, output)
    return fig
