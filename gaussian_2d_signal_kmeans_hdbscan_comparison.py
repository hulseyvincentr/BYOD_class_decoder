#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gaussian_2d_signal_kmeans_hdbscan_comparison.py

Generate two Gaussian clusters with informative signal in the first 2 dimensions
and pure noise in the remaining dimensions. Compare:

1. KMeans on raw data
2. HDBSCAN on raw data
3. HDBSCAN after PCA to 2D
"""

from __future__ import annotations

from typing import Sequence

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


def build_noise_std_vector(
    total_dim: int,
    noise_stds: float | Sequence[float] = 1.0,
) -> np.ndarray:
    if total_dim < 2:
        raise ValueError("total_dim must be >= 2")

    n_noise_dims = total_dim - 2

    if np.isscalar(noise_stds):
        return np.full(n_noise_dims, float(noise_stds), dtype=float)

    noise_std_vector = np.asarray(noise_stds, dtype=float)

    if len(noise_std_vector) != n_noise_dims:
        raise ValueError(
            f"When noise_stds is array-like, it must have length total_dim - 2 = {n_noise_dims}"
        )

    return noise_std_vector


def make_noise_profile(
    total_dim: int,
    profile: str = "constant",
    min_std: float = 1.0,
    max_std: float = 1.0,
) -> np.ndarray:
    if total_dim < 2:
        raise ValueError("total_dim must be >= 2")

    n_noise_dims = total_dim - 2

    if n_noise_dims == 0:
        return np.array([], dtype=float)

    if profile == "constant":
        return np.full(n_noise_dims, float(min_std), dtype=float)

    if profile == "linear":
        return np.linspace(min_std, max_std, n_noise_dims)

    if profile == "geometric":
        if min_std <= 0 or max_std <= 0:
            raise ValueError("min_std and max_std must be > 0 for geometric profile")
        return np.geomspace(min_std, max_std, n_noise_dims)

    raise ValueError('profile must be "constant", "linear", or "geometric"')


def generate_two_gaussian_clusters_with_noise(
    n_per_cluster: int = 100,
    total_dim: int = 10,
    mean_shift: float = 2.0,
    informative_std: float = 1.0,
    noise_stds: float | Sequence[float] = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if total_dim < 2:
        raise ValueError("total_dim must be >= 2")

    rng = np.random.default_rng(random_state)
    noise_std_vector = build_noise_std_vector(total_dim=total_dim, noise_stds=noise_stds)

    n_total = 2 * n_per_cluster
    X = np.zeros((n_total, total_dim), dtype=float)
    y = np.array([0] * n_per_cluster + [1] * n_per_cluster)

    half_shift = mean_shift / 2.0
    mean0 = np.array([-half_shift, -half_shift], dtype=float)
    mean1 = np.array([half_shift, half_shift], dtype=float)

    X[:n_per_cluster, :2] = rng.normal(
        loc=mean0,
        scale=informative_std,
        size=(n_per_cluster, 2),
    )
    X[n_per_cluster:, :2] = rng.normal(
        loc=mean1,
        scale=informative_std,
        size=(n_per_cluster, 2),
    )

    if total_dim > 2:
        noise_block = rng.normal(
            loc=0.0,
            scale=noise_std_vector,
            size=(n_total, total_dim - 2),
        )
        X[:, 2:] = noise_block

    return X, y, noise_std_vector


def between_within_distance_ratio(X: np.ndarray, y: np.ndarray) -> float:
    dist_matrix = squareform(pdist(X, metric="euclidean"))

    same_mask = y[:, None] == y[None, :]
    diff_mask = ~same_mask
    upper_triangle = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)

    within_vals = dist_matrix[same_mask & upper_triangle]
    between_vals = dist_matrix[diff_mask & upper_triangle]

    return float(np.mean(between_vals) / np.mean(within_vals))


def summarize_hdbscan_labels(labels: np.ndarray) -> dict:
    clustered_fraction = float(np.mean(labels != -1))
    unique_clusters = np.unique(labels[labels != -1])
    n_clusters_found = int(len(unique_clusters))

    return {
        "clustered_fraction": clustered_fraction,
        "n_clusters_found": n_clusters_found,
    }


def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
) -> np.ndarray:
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    return model.fit_predict(X)


def evaluate_methods(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
    kmeans_n_init: int = 10,
    pca_components: int = 2,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
) -> dict:
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=kmeans_n_init)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_ari = adjusted_rand_score(y, kmeans_labels)

    raw_hdbscan_labels = fit_hdbscan(
        X,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        allow_single_cluster=hdbscan_allow_single_cluster,
    )
    raw_hdbscan_ari = adjusted_rand_score(y, raw_hdbscan_labels)
    raw_hdbscan_summary = summarize_hdbscan_labels(raw_hdbscan_labels)

    n_components = min(pca_components, X.shape[1])
    X_pca = PCA(n_components=n_components).fit_transform(X)

    pca_hdbscan_labels = fit_hdbscan(
        X_pca,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        allow_single_cluster=hdbscan_allow_single_cluster,
    )
    pca_hdbscan_ari = adjusted_rand_score(y, pca_hdbscan_labels)
    pca_hdbscan_summary = summarize_hdbscan_labels(pca_hdbscan_labels)

    results = {
        "kmeans_ari": float(kmeans_ari),
        "raw_hdbscan_ari": float(raw_hdbscan_ari),
        "pca_hdbscan_ari": float(pca_hdbscan_ari),
        "raw_hdbscan_clustered_fraction": float(raw_hdbscan_summary["clustered_fraction"]),
        "pca_hdbscan_clustered_fraction": float(pca_hdbscan_summary["clustered_fraction"]),
        "raw_hdbscan_n_clusters": int(raw_hdbscan_summary["n_clusters_found"]),
        "pca_hdbscan_n_clusters": int(pca_hdbscan_summary["n_clusters_found"]),
        "raw_data_sep_ratio": float(between_within_distance_ratio(X, y)),
        "pca_data_sep_ratio": float(between_within_distance_ratio(X_pca, y)),
        "kmeans_labels": kmeans_labels,
        "raw_hdbscan_labels": raw_hdbscan_labels,
        "pca_hdbscan_labels": pca_hdbscan_labels,
        "X_pca": X_pca,
    }

    return results


def _plot_truth(ax, coords, y, title, xlabel, ylabel, point_size=30, alpha=0.8):
    for lab in np.unique(y):
        mask = y == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"True cluster {lab}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def _plot_labels(ax, coords, labels, title, xlabel, ylabel, point_size=30, alpha=0.85):
    unique_labels = np.unique(labels)
    non_noise_labels = [lab for lab in unique_labels if lab != -1]

    for lab in non_noise_labels:
        mask = labels == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"Cluster {lab}",
        )

    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(
            coords[noise_mask, 0],
            coords[noise_mask, 1],
            s=point_size,
            alpha=alpha,
            color="gray",
            marker="x",
            label="Noise (-1)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_method_comparison_figure(
    X: np.ndarray,
    y: np.ndarray,
    kmeans_labels: np.ndarray,
    raw_hdbscan_labels: np.ndarray,
    pca_hdbscan_labels: np.ndarray,
    X_pca: np.ndarray,
    kmeans_ari: float | None = None,
    raw_hdbscan_ari: float | None = None,
    pca_hdbscan_ari: float | None = None,
    raw_hdbscan_clustered_fraction: float | None = None,
    pca_hdbscan_clustered_fraction: float | None = None,
    raw_hdbscan_n_clusters: int | None = None,
    pca_hdbscan_n_clusters: int | None = None,
    total_dim: int | None = None,
    point_size: int = 30,
    alpha: float = 0.85,
    show: bool = True,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    dim_text = f"total_dim={total_dim}" if total_dim is not None else "data"

    _plot_truth(
        axes[0, 0],
        X[:, :2],
        y,
        title=f"True labels in informative 2D ({dim_text})",
        xlabel="Informative dim 1",
        ylabel="Informative dim 2",
        point_size=point_size,
        alpha=alpha,
    )

    kmeans_title = "KMeans on raw data"
    if kmeans_ari is not None:
        kmeans_title += f"\nARI={kmeans_ari:.3f}"

    _plot_labels(
        axes[0, 1],
        X[:, :2],
        kmeans_labels,
        title=kmeans_title,
        xlabel="Informative dim 1",
        ylabel="Informative dim 2",
        point_size=point_size,
        alpha=alpha,
    )

    raw_hdb_title = "HDBSCAN on raw data"
    raw_metrics = []
    if raw_hdbscan_ari is not None:
        raw_metrics.append(f"ARI={raw_hdbscan_ari:.3f}")
    if raw_hdbscan_clustered_fraction is not None:
        raw_metrics.append(f"clustered={raw_hdbscan_clustered_fraction:.2f}")
    if raw_hdbscan_n_clusters is not None:
        raw_metrics.append(f"clusters={raw_hdbscan_n_clusters}")
    if raw_metrics:
        raw_hdb_title += "\n" + ", ".join(raw_metrics)

    _plot_labels(
        axes[0, 2],
        X[:, :2],
        raw_hdbscan_labels,
        title=raw_hdb_title,
        xlabel="Informative dim 1",
        ylabel="Informative dim 2",
        point_size=point_size,
        alpha=alpha,
    )

    _plot_truth(
        axes[1, 0],
        X_pca,
        y,
        title="True labels in PCA 2D",
        xlabel="PC1",
        ylabel="PC2",
        point_size=point_size,
        alpha=alpha,
    )

    pca_hdb_title = "HDBSCAN after PCA to 2D"
    pca_metrics = []
    if pca_hdbscan_ari is not None:
        pca_metrics.append(f"ARI={pca_hdbscan_ari:.3f}")
    if pca_hdbscan_clustered_fraction is not None:
        pca_metrics.append(f"clustered={pca_hdbscan_clustered_fraction:.2f}")
    if pca_hdbscan_n_clusters is not None:
        pca_metrics.append(f"clusters={pca_hdbscan_n_clusters}")
    if pca_metrics:
        pca_hdb_title += "\n" + ", ".join(pca_metrics)

    _plot_labels(
        axes[1, 1],
        X_pca,
        pca_hdbscan_labels,
        title=pca_hdb_title,
        xlabel="PC1",
        ylabel="PC2",
        point_size=point_size,
        alpha=alpha,
    )

    axes[1, 2].axis("off")
    summary_lines = [
        "Summary",
        "-------",
    ]
    if kmeans_ari is not None:
        summary_lines.append(f"KMeans ARI: {kmeans_ari:.3f}")
    if raw_hdbscan_ari is not None:
        summary_lines.append(f"Raw HDBSCAN ARI: {raw_hdbscan_ari:.3f}")
    if raw_hdbscan_clustered_fraction is not None:
        summary_lines.append(f"Raw HDBSCAN clustered: {raw_hdbscan_clustered_fraction:.2f}")
    if raw_hdbscan_n_clusters is not None:
        summary_lines.append(f"Raw HDBSCAN clusters: {raw_hdbscan_n_clusters}")
    if pca_hdbscan_ari is not None:
        summary_lines.append(f"PCA+HDBSCAN ARI: {pca_hdbscan_ari:.3f}")
    if pca_hdbscan_clustered_fraction is not None:
        summary_lines.append(f"PCA+HDBSCAN clustered: {pca_hdbscan_clustered_fraction:.2f}")
    if pca_hdbscan_n_clusters is not None:
        summary_lines.append(f"PCA+HDBSCAN clusters: {pca_hdbscan_n_clusters}")

    axes[1, 2].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=13,
        family="monospace",
    )

    fig.tight_layout()

    if show:
        plt.show()


if __name__ == "__main__":
    total_dim = 100

    noise_std_vector = make_noise_profile(
        total_dim=total_dim,
        profile="linear",
        min_std=0.5,
        max_std=3.0,
    )

    X, y, noise_std_vector = generate_two_gaussian_clusters_with_noise(
        n_per_cluster=100,
        total_dim=total_dim,
        mean_shift=2.0,
        informative_std=1.0,
        noise_stds=noise_std_vector,
        random_state=0,
    )

    results = evaluate_methods(
        X=X,
        y=y,
        random_state=0,
        kmeans_n_init=10,
        pca_components=2,
        hdbscan_min_cluster_size=20,
        hdbscan_min_samples=None,
        hdbscan_metric="euclidean",
        hdbscan_cluster_selection_method="eom",
        hdbscan_allow_single_cluster=False,
    )

    print("Single dataset summary:")
    for key, value in results.items():
        if key in {"kmeans_labels", "raw_hdbscan_labels", "pca_hdbscan_labels", "X_pca"}:
            continue
        print(f"{key}: {value}")

    plot_method_comparison_figure(
        X=X,
        y=y,
        kmeans_labels=results["kmeans_labels"],
        raw_hdbscan_labels=results["raw_hdbscan_labels"],
        pca_hdbscan_labels=results["pca_hdbscan_labels"],
        X_pca=results["X_pca"],
        kmeans_ari=results["kmeans_ari"],
        raw_hdbscan_ari=results["raw_hdbscan_ari"],
        pca_hdbscan_ari=results["pca_hdbscan_ari"],
        raw_hdbscan_clustered_fraction=results["raw_hdbscan_clustered_fraction"],
        pca_hdbscan_clustered_fraction=results["pca_hdbscan_clustered_fraction"],
        raw_hdbscan_n_clusters=results["raw_hdbscan_n_clusters"],
        pca_hdbscan_n_clusters=results["pca_hdbscan_n_clusters"],
        total_dim=total_dim,
        show=True,
    )
