#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gaussian_2d_signal_with_noise_hdbscan_raw_vs_pca.py

Generate two Gaussian clusters with informative signal in the first 2 dimensions
and pure noise in the remaining dimensions. Compare:

1. HDBSCAN on the full raw data
2. HDBSCAN after PCA to 2D

This script makes the results easier to interpret by showing:
- the true cluster labels in informative 2D space
- HDBSCAN assignments from raw high-dimensional data, displayed in informative 2D
- the true labels in PCA 2D space
- HDBSCAN assignments after PCA to 2D, displayed in PCA 2D
"""

from __future__ import annotations

from typing import Sequence

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
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
    labels = model.fit_predict(X)
    return labels


def evaluate_hdbscan_raw_and_pca(
    X: np.ndarray,
    y: np.ndarray,
    pca_components: int = 2,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
) -> dict:
    raw_labels = fit_hdbscan(
        X,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        allow_single_cluster=hdbscan_allow_single_cluster,
    )

    n_components = min(pca_components, X.shape[1])
    X_pca = PCA(n_components=n_components).fit_transform(X)

    pca_labels = fit_hdbscan(
        X_pca,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        allow_single_cluster=hdbscan_allow_single_cluster,
    )

    raw_summary = summarize_hdbscan_labels(raw_labels)
    pca_summary = summarize_hdbscan_labels(pca_labels)

    results = {
        "raw_hdbscan_ari": float(adjusted_rand_score(y, raw_labels)),
        "pca_hdbscan_ari": float(adjusted_rand_score(y, pca_labels)),
        "raw_clustered_fraction": float(raw_summary["clustered_fraction"]),
        "pca_clustered_fraction": float(pca_summary["clustered_fraction"]),
        "raw_n_clusters": int(raw_summary["n_clusters_found"]),
        "pca_n_clusters": int(pca_summary["n_clusters_found"]),
        "raw_data_sep_ratio": float(between_within_distance_ratio(X, y)),
        "pca_data_sep_ratio": float(between_within_distance_ratio(X_pca, y)),
        "raw_labels": raw_labels,
        "pca_labels": pca_labels,
        "X_pca": X_pca,
    }

    return results


def _plot_truth_in_informative_space(
    ax,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    point_size: int = 30,
    alpha: float = 0.8,
):
    for lab in np.unique(y):
        mask = y == lab
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"True cluster {lab}",
        )

    ax.set_xlabel("Informative dim 1")
    ax.set_ylabel("Informative dim 2")
    ax.set_title(title)
    ax.legend()


def _plot_truth_in_pca_space(
    ax,
    X_pca: np.ndarray,
    y: np.ndarray,
    title: str,
    point_size: int = 30,
    alpha: float = 0.8,
):
    for lab in np.unique(y):
        mask = y == lab
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"True cluster {lab}",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()


def _plot_hdbscan_labels(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    point_size: int = 30,
    alpha: float = 0.85,
):
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


def plot_hdbscan_raw_vs_pca_comparison(
    X: np.ndarray,
    y: np.ndarray,
    raw_labels: np.ndarray,
    pca_labels: np.ndarray,
    X_pca: np.ndarray,
    raw_hdbscan_ari: float | None = None,
    pca_hdbscan_ari: float | None = None,
    raw_clustered_fraction: float | None = None,
    pca_clustered_fraction: float | None = None,
    raw_n_clusters: int | None = None,
    pca_n_clusters: int | None = None,
    total_dim: int | None = None,
    point_size: int = 30,
    alpha: float = 0.85,
    title_fontsize: int = 14,
    show: bool = True,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    dim_text = f"total_dim={total_dim}" if total_dim is not None else "data"

    _plot_truth_in_informative_space(
        axes[0, 0],
        X,
        y,
        title=f"True labels in informative 2D ({dim_text})",
        point_size=point_size,
        alpha=alpha,
    )

    raw_title = "HDBSCAN on raw data"
    raw_metrics = []
    if raw_hdbscan_ari is not None:
        raw_metrics.append(f"ARI={raw_hdbscan_ari:.3f}")
    if raw_clustered_fraction is not None:
        raw_metrics.append(f"clustered={raw_clustered_fraction:.2f}")
    if raw_n_clusters is not None:
        raw_metrics.append(f"clusters={raw_n_clusters}")
    if raw_metrics:
        raw_title += "\n" + ", ".join(raw_metrics)

    _plot_hdbscan_labels(
        axes[0, 1],
        X[:, :2],
        raw_labels,
        title=raw_title,
        xlabel="Informative dim 1",
        ylabel="Informative dim 2",
        point_size=point_size,
        alpha=alpha,
    )

    _plot_truth_in_pca_space(
        axes[1, 0],
        X_pca,
        y,
        title="True labels in PCA 2D",
        point_size=point_size,
        alpha=alpha,
    )

    pca_title = "HDBSCAN after PCA to 2D"
    pca_metrics = []
    if pca_hdbscan_ari is not None:
        pca_metrics.append(f"ARI={pca_hdbscan_ari:.3f}")
    if pca_clustered_fraction is not None:
        pca_metrics.append(f"clustered={pca_clustered_fraction:.2f}")
    if pca_n_clusters is not None:
        pca_metrics.append(f"clusters={pca_n_clusters}")
    if pca_metrics:
        pca_title += "\n" + ", ".join(pca_metrics)

    _plot_hdbscan_labels(
        axes[1, 1],
        X_pca,
        pca_labels,
        title=pca_title,
        xlabel="PC1",
        ylabel="PC2",
        point_size=point_size,
        alpha=alpha,
    )

    for ax in axes.ravel():
        ax.tick_params(labelsize=11)
        ax.title.set_fontsize(title_fontsize)

    fig.tight_layout()

    if show:
        plt.show()


def sweep_total_dimension(
    total_dims: Sequence[int],
    n_per_cluster: int = 100,
    mean_shift: float = 2.0,
    informative_std: float = 1.0,
    noise_stds: float | Sequence[float] = 1.0,
    n_repeats: int = 10,
    base_seed: int = 0,
    pca_components: int = 2,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
) -> pd.DataFrame:
    rows = []

    for total_dim in total_dims:
        raw_ari_values = []
        pca_ari_values = []
        raw_clustered_fraction_values = []
        pca_clustered_fraction_values = []
        raw_n_clusters_values = []
        pca_n_clusters_values = []
        raw_sep_ratio_values = []
        pca_sep_ratio_values = []

        for r in range(n_repeats):
            seed = base_seed + 1000 * r + total_dim

            if np.isscalar(noise_stds):
                current_noise_stds = noise_stds
            else:
                current_noise_stds = np.asarray(noise_stds[: max(total_dim - 2, 0)], dtype=float)

            X, y, _ = generate_two_gaussian_clusters_with_noise(
                n_per_cluster=n_per_cluster,
                total_dim=total_dim,
                mean_shift=mean_shift,
                informative_std=informative_std,
                noise_stds=current_noise_stds,
                random_state=seed,
            )

            results = evaluate_hdbscan_raw_and_pca(
                X=X,
                y=y,
                pca_components=pca_components,
                hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                hdbscan_min_samples=hdbscan_min_samples,
                hdbscan_metric=hdbscan_metric,
                hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
                hdbscan_allow_single_cluster=hdbscan_allow_single_cluster,
            )

            raw_ari_values.append(results["raw_hdbscan_ari"])
            pca_ari_values.append(results["pca_hdbscan_ari"])
            raw_clustered_fraction_values.append(results["raw_clustered_fraction"])
            pca_clustered_fraction_values.append(results["pca_clustered_fraction"])
            raw_n_clusters_values.append(results["raw_n_clusters"])
            pca_n_clusters_values.append(results["pca_n_clusters"])
            raw_sep_ratio_values.append(results["raw_data_sep_ratio"])
            pca_sep_ratio_values.append(results["pca_data_sep_ratio"])

        rows.append({
            "total_dim": int(total_dim),
            "raw_hdbscan_ari_mean": float(np.mean(raw_ari_values)),
            "raw_hdbscan_ari_sd": float(np.std(raw_ari_values, ddof=1)),
            "pca_hdbscan_ari_mean": float(np.mean(pca_ari_values)),
            "pca_hdbscan_ari_sd": float(np.std(pca_ari_values, ddof=1)),
            "raw_clustered_fraction_mean": float(np.mean(raw_clustered_fraction_values)),
            "raw_clustered_fraction_sd": float(np.std(raw_clustered_fraction_values, ddof=1)),
            "pca_clustered_fraction_mean": float(np.mean(pca_clustered_fraction_values)),
            "pca_clustered_fraction_sd": float(np.std(pca_clustered_fraction_values, ddof=1)),
            "raw_n_clusters_mean": float(np.mean(raw_n_clusters_values)),
            "raw_n_clusters_sd": float(np.std(raw_n_clusters_values, ddof=1)),
            "pca_n_clusters_mean": float(np.mean(pca_n_clusters_values)),
            "pca_n_clusters_sd": float(np.std(pca_n_clusters_values, ddof=1)),
            "raw_data_sep_ratio_mean": float(np.mean(raw_sep_ratio_values)),
            "raw_data_sep_ratio_sd": float(np.std(raw_sep_ratio_values, ddof=1)),
            "pca_data_sep_ratio_mean": float(np.mean(pca_sep_ratio_values)),
            "pca_data_sep_ratio_sd": float(np.std(pca_sep_ratio_values, ddof=1)),
        })

    return pd.DataFrame(rows)


def sweep_noise_std(
    noise_std_values: Sequence[float],
    n_per_cluster: int = 100,
    total_dim: int = 100,
    mean_shift: float = 2.0,
    informative_std: float = 1.0,
    n_repeats: int = 10,
    base_seed: int = 0,
    pca_components: int = 2,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
) -> pd.DataFrame:
    rows = []

    for noise_std in noise_std_values:
        raw_ari_values = []
        pca_ari_values = []
        raw_clustered_fraction_values = []
        pca_clustered_fraction_values = []
        raw_n_clusters_values = []
        pca_n_clusters_values = []
        raw_sep_ratio_values = []
        pca_sep_ratio_values = []

        for r in range(n_repeats):
            seed = base_seed + 1000 * r + int(100 * noise_std)

            X, y, _ = generate_two_gaussian_clusters_with_noise(
                n_per_cluster=n_per_cluster,
                total_dim=total_dim,
                mean_shift=mean_shift,
                informative_std=informative_std,
                noise_stds=noise_std,
                random_state=seed,
            )

            results = evaluate_hdbscan_raw_and_pca(
                X=X,
                y=y,
                pca_components=pca_components,
                hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                hdbscan_min_samples=hdbscan_min_samples,
                hdbscan_metric=hdbscan_metric,
                hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
                hdbscan_allow_single_cluster=hdbscan_allow_single_cluster,
            )

            raw_ari_values.append(results["raw_hdbscan_ari"])
            pca_ari_values.append(results["pca_hdbscan_ari"])
            raw_clustered_fraction_values.append(results["raw_clustered_fraction"])
            pca_clustered_fraction_values.append(results["pca_clustered_fraction"])
            raw_n_clusters_values.append(results["raw_n_clusters"])
            pca_n_clusters_values.append(results["pca_n_clusters"])
            raw_sep_ratio_values.append(results["raw_data_sep_ratio"])
            pca_sep_ratio_values.append(results["pca_data_sep_ratio"])

        rows.append({
            "noise_std": float(noise_std),
            "total_dim": int(total_dim),
            "raw_hdbscan_ari_mean": float(np.mean(raw_ari_values)),
            "raw_hdbscan_ari_sd": float(np.std(raw_ari_values, ddof=1)),
            "pca_hdbscan_ari_mean": float(np.mean(pca_ari_values)),
            "pca_hdbscan_ari_sd": float(np.std(pca_ari_values, ddof=1)),
            "raw_clustered_fraction_mean": float(np.mean(raw_clustered_fraction_values)),
            "raw_clustered_fraction_sd": float(np.std(raw_clustered_fraction_values, ddof=1)),
            "pca_clustered_fraction_mean": float(np.mean(pca_clustered_fraction_values)),
            "pca_clustered_fraction_sd": float(np.std(pca_clustered_fraction_values, ddof=1)),
            "raw_n_clusters_mean": float(np.mean(raw_n_clusters_values)),
            "raw_n_clusters_sd": float(np.std(raw_n_clusters_values, ddof=1)),
            "pca_n_clusters_mean": float(np.mean(pca_n_clusters_values)),
            "pca_n_clusters_sd": float(np.std(pca_n_clusters_values, ddof=1)),
            "raw_data_sep_ratio_mean": float(np.mean(raw_sep_ratio_values)),
            "raw_data_sep_ratio_sd": float(np.std(raw_sep_ratio_values, ddof=1)),
            "pca_data_sep_ratio_mean": float(np.mean(pca_sep_ratio_values)),
            "pca_data_sep_ratio_sd": float(np.std(pca_sep_ratio_values, ddof=1)),
        })

    return pd.DataFrame(rows)


def plot_dimension_sweep(
    summary_df: pd.DataFrame,
    metric: str = "ari",
    title: str | None = None,
    show: bool = True,
):
    x = summary_df["total_dim"].to_numpy()

    plt.figure(figsize=(7, 5))

    if metric == "ari":
        plt.plot(x, summary_df["raw_hdbscan_ari_mean"], marker="o", linewidth=2, label="HDBSCAN on raw data")
        plt.plot(x, summary_df["pca_hdbscan_ari_mean"], marker="s", linewidth=2, label="HDBSCAN after PCA to 2D")
        ylabel = "Adjusted Rand Index"
        if title is None:
            title = "HDBSCAN performance vs total dimension"
    elif metric == "clustered_fraction":
        plt.plot(x, summary_df["raw_clustered_fraction_mean"], marker="o", linewidth=2, label="Raw clustered fraction")
        plt.plot(x, summary_df["pca_clustered_fraction_mean"], marker="s", linewidth=2, label="PCA clustered fraction")
        ylabel = "Fraction not labeled noise"
        if title is None:
            title = "HDBSCAN clustered fraction vs total dimension"
    elif metric == "n_clusters":
        plt.plot(x, summary_df["raw_n_clusters_mean"], marker="o", linewidth=2, label="Raw clusters found")
        plt.plot(x, summary_df["pca_n_clusters_mean"], marker="s", linewidth=2, label="PCA clusters found")
        ylabel = "Number of clusters found"
        if title is None:
            title = "HDBSCAN clusters found vs total dimension"
    elif metric == "sep_ratio":
        plt.plot(x, summary_df["raw_data_sep_ratio_mean"], marker="o", linewidth=2, label="Raw between/within ratio")
        plt.plot(x, summary_df["pca_data_sep_ratio_mean"], marker="s", linewidth=2, label="PCA between/within ratio")
        ylabel = "Mean between / within distance"
        if title is None:
            title = "Separation vs total dimension"
    else:
        raise ValueError('metric must be "ari", "clustered_fraction", "n_clusters", or "sep_ratio"')

    plt.xlabel("Total dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()


def plot_noise_std_sweep(
    summary_df: pd.DataFrame,
    metric: str = "ari",
    title: str | None = None,
    show: bool = True,
):
    x = summary_df["noise_std"].to_numpy()

    plt.figure(figsize=(7, 5))

    if metric == "ari":
        plt.plot(x, summary_df["raw_hdbscan_ari_mean"], marker="o", linewidth=2, label="HDBSCAN on raw data")
        plt.plot(x, summary_df["pca_hdbscan_ari_mean"], marker="s", linewidth=2, label="HDBSCAN after PCA to 2D")
        ylabel = "Adjusted Rand Index"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"HDBSCAN performance vs noise SD (total_dim={total_dim})"
    elif metric == "clustered_fraction":
        plt.plot(x, summary_df["raw_clustered_fraction_mean"], marker="o", linewidth=2, label="Raw clustered fraction")
        plt.plot(x, summary_df["pca_clustered_fraction_mean"], marker="s", linewidth=2, label="PCA clustered fraction")
        ylabel = "Fraction not labeled noise"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"HDBSCAN clustered fraction vs noise SD (total_dim={total_dim})"
    elif metric == "n_clusters":
        plt.plot(x, summary_df["raw_n_clusters_mean"], marker="o", linewidth=2, label="Raw clusters found")
        plt.plot(x, summary_df["pca_n_clusters_mean"], marker="s", linewidth=2, label="PCA clusters found")
        ylabel = "Number of clusters found"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"HDBSCAN clusters found vs noise SD (total_dim={total_dim})"
    elif metric == "sep_ratio":
        plt.plot(x, summary_df["raw_data_sep_ratio_mean"], marker="o", linewidth=2, label="Raw between/within ratio")
        plt.plot(x, summary_df["pca_data_sep_ratio_mean"], marker="s", linewidth=2, label="PCA between/within ratio")
        ylabel = "Mean between / within distance"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"Separation vs noise SD (total_dim={total_dim})"
    else:
        raise ValueError('metric must be "ari", "clustered_fraction", "n_clusters", or "sep_ratio"')

    plt.xlabel("Noise standard deviation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

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

    results = evaluate_hdbscan_raw_and_pca(
        X=X,
        y=y,
        pca_components=2,
        hdbscan_min_cluster_size=20,
        hdbscan_min_samples=None,
        hdbscan_metric="euclidean",
        hdbscan_cluster_selection_method="eom",
        hdbscan_allow_single_cluster=False,
    )

    print("Single dataset summary:")
    for key, value in results.items():
        if key in {"raw_labels", "pca_labels", "X_pca"}:
            continue
        print(f"{key}: {value}")

    plot_hdbscan_raw_vs_pca_comparison(
        X=X,
        y=y,
        raw_labels=results["raw_labels"],
        pca_labels=results["pca_labels"],
        X_pca=results["X_pca"],
        raw_hdbscan_ari=results["raw_hdbscan_ari"],
        pca_hdbscan_ari=results["pca_hdbscan_ari"],
        raw_clustered_fraction=results["raw_clustered_fraction"],
        pca_clustered_fraction=results["pca_clustered_fraction"],
        raw_n_clusters=results["raw_n_clusters"],
        pca_n_clusters=results["pca_n_clusters"],
        total_dim=total_dim,
        show=True,
    )
