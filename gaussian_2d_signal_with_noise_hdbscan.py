#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gaussian_2d_signal_with_noise_hdbscan.py

Generate two Gaussian clusters with informative signal in the first 2 dimensions
and pure noise in the remaining dimensions, where the noise dimensions can have
user-defined standard deviations.

This version compares:
- KMeans on the full data
- HDBSCAN on the full data

It also tracks HDBSCAN-specific diagnostics such as:
- fraction of points labeled as noise
- number of clusters found

Dependencies
------------
numpy
pandas
matplotlib
scipy
scikit-learn
hdbscan
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import hdbscan


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


def evaluate_kmeans_and_hdbscan(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
    kmeans_n_init: int = 10,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
) -> dict:
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=kmeans_n_init)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_ari = adjusted_rand_score(y, kmeans_labels)

    hdb = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        allow_single_cluster=hdbscan_allow_single_cluster,
    )
    hdb_labels = hdb.fit_predict(X)

    hdbscan_ari = adjusted_rand_score(y, hdb_labels)
    clustered_fraction = float(np.mean(hdb_labels != -1))
    unique_clusters = np.unique(hdb_labels[hdb_labels != -1])
    n_clusters_found = int(len(unique_clusters))

    sep_ratio = between_within_distance_ratio(X, y)

    return {
        "kmeans_ari": float(kmeans_ari),
        "hdbscan_ari": float(hdbscan_ari),
        "hdbscan_clustered_fraction": clustered_fraction,
        "hdbscan_n_clusters": n_clusters_found,
        "data_sep_ratio": float(sep_ratio),
        "kmeans_labels": kmeans_labels,
        "hdbscan_labels": hdb_labels,
    }


def sweep_total_dimension(
    total_dims: Sequence[int],
    n_per_cluster: int = 100,
    mean_shift: float = 2.0,
    informative_std: float = 1.0,
    noise_stds: float | Sequence[float] = 1.0,
    n_repeats: int = 10,
    base_seed: int = 0,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
) -> pd.DataFrame:
    rows = []

    for total_dim in total_dims:
        kmeans_ari_values = []
        hdbscan_ari_values = []
        hdbscan_clustered_fraction_values = []
        hdbscan_n_clusters_values = []
        sep_ratio_values = []

        for r in range(n_repeats):
            seed = base_seed + 1000 * r + total_dim

            if np.isscalar(noise_stds):
                current_noise_stds = noise_stds
            else:
                current_noise_stds = np.asarray(noise_stds[: max(total_dim - 2, 0)], dtype=float)

            X, y, noise_std_vector = generate_two_gaussian_clusters_with_noise(
                n_per_cluster=n_per_cluster,
                total_dim=total_dim,
                mean_shift=mean_shift,
                informative_std=informative_std,
                noise_stds=current_noise_stds,
                random_state=seed,
            )

            metrics = evaluate_kmeans_and_hdbscan(
                X=X,
                y=y,
                random_state=seed,
                hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                hdbscan_min_samples=hdbscan_min_samples,
                hdbscan_metric=hdbscan_metric,
                hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
                hdbscan_allow_single_cluster=hdbscan_allow_single_cluster,
            )

            kmeans_ari_values.append(metrics["kmeans_ari"])
            hdbscan_ari_values.append(metrics["hdbscan_ari"])
            hdbscan_clustered_fraction_values.append(metrics["hdbscan_clustered_fraction"])
            hdbscan_n_clusters_values.append(metrics["hdbscan_n_clusters"])
            sep_ratio_values.append(metrics["data_sep_ratio"])

        row = {
            "total_dim": int(total_dim),
            "kmeans_ari_mean": float(np.mean(kmeans_ari_values)),
            "kmeans_ari_sd": float(np.std(kmeans_ari_values, ddof=1)),
            "hdbscan_ari_mean": float(np.mean(hdbscan_ari_values)),
            "hdbscan_ari_sd": float(np.std(hdbscan_ari_values, ddof=1)),
            "hdbscan_clustered_fraction_mean": float(np.mean(hdbscan_clustered_fraction_values)),
            "hdbscan_clustered_fraction_sd": float(np.std(hdbscan_clustered_fraction_values, ddof=1)),
            "hdbscan_n_clusters_mean": float(np.mean(hdbscan_n_clusters_values)),
            "hdbscan_n_clusters_sd": float(np.std(hdbscan_n_clusters_values, ddof=1)),
            "data_sep_ratio_mean": float(np.mean(sep_ratio_values)),
            "data_sep_ratio_sd": float(np.std(sep_ratio_values, ddof=1)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def sweep_noise_std(
    noise_std_values: Sequence[float],
    n_per_cluster: int = 100,
    total_dim: int = 100,
    mean_shift: float = 2.0,
    informative_std: float = 1.0,
    n_repeats: int = 10,
    base_seed: int = 0,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int | None = None,
    hdbscan_metric: str = "euclidean",
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
) -> pd.DataFrame:
    rows = []

    for noise_std in noise_std_values:
        kmeans_ari_values = []
        hdbscan_ari_values = []
        hdbscan_clustered_fraction_values = []
        hdbscan_n_clusters_values = []
        sep_ratio_values = []

        for r in range(n_repeats):
            seed = base_seed + 1000 * r + int(100 * noise_std)

            X, y, noise_std_vector = generate_two_gaussian_clusters_with_noise(
                n_per_cluster=n_per_cluster,
                total_dim=total_dim,
                mean_shift=mean_shift,
                informative_std=informative_std,
                noise_stds=noise_std,
                random_state=seed,
            )

            metrics = evaluate_kmeans_and_hdbscan(
                X=X,
                y=y,
                random_state=seed,
                hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                hdbscan_min_samples=hdbscan_min_samples,
                hdbscan_metric=hdbscan_metric,
                hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
                hdbscan_allow_single_cluster=hdbscan_allow_single_cluster,
            )

            kmeans_ari_values.append(metrics["kmeans_ari"])
            hdbscan_ari_values.append(metrics["hdbscan_ari"])
            hdbscan_clustered_fraction_values.append(metrics["hdbscan_clustered_fraction"])
            hdbscan_n_clusters_values.append(metrics["hdbscan_n_clusters"])
            sep_ratio_values.append(metrics["data_sep_ratio"])

        row = {
            "noise_std": float(noise_std),
            "total_dim": int(total_dim),
            "kmeans_ari_mean": float(np.mean(kmeans_ari_values)),
            "kmeans_ari_sd": float(np.std(kmeans_ari_values, ddof=1)),
            "hdbscan_ari_mean": float(np.mean(hdbscan_ari_values)),
            "hdbscan_ari_sd": float(np.std(hdbscan_ari_values, ddof=1)),
            "hdbscan_clustered_fraction_mean": float(np.mean(hdbscan_clustered_fraction_values)),
            "hdbscan_clustered_fraction_sd": float(np.std(hdbscan_clustered_fraction_values, ddof=1)),
            "hdbscan_n_clusters_mean": float(np.mean(hdbscan_n_clusters_values)),
            "hdbscan_n_clusters_sd": float(np.std(hdbscan_n_clusters_values, ddof=1)),
            "data_sep_ratio_mean": float(np.mean(sep_ratio_values)),
            "data_sep_ratio_sd": float(np.std(sep_ratio_values, ddof=1)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_informative_2d_view(
    X: np.ndarray,
    y: np.ndarray,
    mean_shift: float | None = None,
    informative_std: float | None = None,
    title: str | None = None,
    point_size: int = 30,
    alpha: float = 0.7,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    tick_fontsize: int = 12,
    title_fontsize: int = 15,
    legend_fontsize: int = 12,
    show: bool = True,
):
    plt.figure(figsize=(7, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=point_size, alpha=alpha, label="True cluster 1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=point_size, alpha=alpha, label="True cluster 2")

    plt.xlabel("Informative dim 1", fontsize=xlabel_fontsize)
    plt.ylabel("Informative dim 2", fontsize=ylabel_fontsize)

    if title is None:
        title = "True informative 2D signal"
        extras = []
        if mean_shift is not None:
            extras.append(f"mean_shift={mean_shift}")
        if informative_std is not None:
            extras.append(f"informative_std={informative_std}")
        if extras:
            title += " (" + ", ".join(extras) + ")"

    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if show:
        plt.show()


def plot_hdbscan_assignments_in_informative_space(
    X: np.ndarray,
    hdbscan_labels: np.ndarray,
    total_dim: int | None = None,
    title: str | None = None,
    point_size: int = 30,
    alpha: float = 0.8,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    tick_fontsize: int = 12,
    title_fontsize: int = 15,
    show: bool = True,
):
    plt.figure(figsize=(7, 6))

    unique_labels = np.unique(hdbscan_labels)
    non_noise_labels = [lab for lab in unique_labels if lab != -1]

    for lab in non_noise_labels:
        mask = hdbscan_labels == lab
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            s=point_size,
            alpha=alpha,
            label=f"HDBSCAN cluster {lab}",
        )

    noise_mask = hdbscan_labels == -1
    if np.any(noise_mask):
        plt.scatter(
            X[noise_mask, 0],
            X[noise_mask, 1],
            s=point_size,
            alpha=alpha,
            color="gray",
            label="Noise (-1)",
        )

    plt.xlabel("Informative dim 1", fontsize=xlabel_fontsize)
    plt.ylabel("Informative dim 2", fontsize=ylabel_fontsize)

    if title is None:
        if total_dim is not None:
            title = f"HDBSCAN assignments in informative 2D space (total_dim={total_dim})"
        else:
            title = "HDBSCAN assignments in informative 2D space"

    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()


def plot_dimension_sweep(
    summary_df: pd.DataFrame,
    metric: str = "ari",
    title: str | None = None,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    tick_fontsize: int = 12,
    title_fontsize: int = 15,
    legend_fontsize: int = 12,
    show: bool = True,
):
    x = summary_df["total_dim"].to_numpy()

    plt.figure(figsize=(7, 5))

    if metric == "ari":
        plt.plot(x, summary_df["kmeans_ari_mean"], marker="o", linewidth=2, label="KMeans ARI")
        plt.plot(x, summary_df["hdbscan_ari_mean"], marker="s", linewidth=2, label="HDBSCAN ARI")
        ylabel = "Adjusted Rand Index"
        if title is None:
            title = "Clustering performance vs total dimension"
    elif metric == "sep_ratio":
        plt.plot(
            x,
            summary_df["data_sep_ratio_mean"],
            marker="o",
            linewidth=2,
            label="Between/within distance ratio",
        )
        ylabel = "Mean between / within distance"
        if title is None:
            title = "Data separation vs total dimension"
    elif metric == "clustered_fraction":
        plt.plot(
            x,
            summary_df["hdbscan_clustered_fraction_mean"],
            marker="s",
            linewidth=2,
            label="HDBSCAN clustered fraction",
        )
        ylabel = "Fraction of points not labeled noise"
        if title is None:
            title = "HDBSCAN clustered fraction vs total dimension"
    elif metric == "n_clusters":
        plt.plot(
            x,
            summary_df["hdbscan_n_clusters_mean"],
            marker="s",
            linewidth=2,
            label="HDBSCAN number of clusters",
        )
        ylabel = "Number of clusters found"
        if title is None:
            title = "HDBSCAN clusters found vs total dimension"
    else:
        raise ValueError('metric must be "ari", "sep_ratio", "clustered_fraction", or "n_clusters"')

    plt.xlabel("Total dimension", fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if show:
        plt.show()


def plot_noise_std_sweep(
    summary_df: pd.DataFrame,
    metric: str = "ari",
    title: str | None = None,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    tick_fontsize: int = 12,
    title_fontsize: int = 15,
    legend_fontsize: int = 12,
    show: bool = True,
):
    x = summary_df["noise_std"].to_numpy()

    plt.figure(figsize=(7, 5))

    if metric == "ari":
        plt.plot(x, summary_df["kmeans_ari_mean"], marker="o", linewidth=2, label="KMeans ARI")
        plt.plot(x, summary_df["hdbscan_ari_mean"], marker="s", linewidth=2, label="HDBSCAN ARI")
        ylabel = "Adjusted Rand Index"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"Clustering performance vs noise SD (total_dim={total_dim})"
    elif metric == "sep_ratio":
        plt.plot(
            x,
            summary_df["data_sep_ratio_mean"],
            marker="o",
            linewidth=2,
            label="Between/within distance ratio",
        )
        ylabel = "Mean between / within distance"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"Data separation vs noise SD (total_dim={total_dim})"
    elif metric == "clustered_fraction":
        plt.plot(
            x,
            summary_df["hdbscan_clustered_fraction_mean"],
            marker="s",
            linewidth=2,
            label="HDBSCAN clustered fraction",
        )
        ylabel = "Fraction of points not labeled noise"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"HDBSCAN clustered fraction vs noise SD (total_dim={total_dim})"
    elif metric == "n_clusters":
        plt.plot(
            x,
            summary_df["hdbscan_n_clusters_mean"],
            marker="s",
            linewidth=2,
            label="HDBSCAN number of clusters",
        )
        ylabel = "Number of clusters found"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"HDBSCAN clusters found vs noise SD (total_dim={total_dim})"
    else:
        raise ValueError('metric must be "ari", "sep_ratio", "clustered_fraction", or "n_clusters"')

    plt.xlabel("Noise standard deviation", fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
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

    print("Noise std vector summary:")
    print("min noise SD:", noise_std_vector.min() if len(noise_std_vector) > 0 else "none")
    print("max noise SD:", noise_std_vector.max() if len(noise_std_vector) > 0 else "none")
    print("number of noise dims:", len(noise_std_vector))

    metrics = evaluate_kmeans_and_hdbscan(
        X=X,
        y=y,
        random_state=0,
        hdbscan_min_cluster_size=20,
        hdbscan_min_samples=None,
        hdbscan_metric="euclidean",
        hdbscan_cluster_selection_method="eom",
        hdbscan_allow_single_cluster=False,
    )

    print("\nSingle dataset metrics:")
    for key, value in metrics.items():
        if key.endswith("_labels"):
            continue
        print(f"{key}: {value}")

    plot_informative_2d_view(
        X,
        y,
        mean_shift=2.0,
        informative_std=1.0,
    )

    plot_hdbscan_assignments_in_informative_space(
        X,
        metrics["hdbscan_labels"],
        total_dim=total_dim,
    )

    noise_summary = sweep_noise_std(
        noise_std_values=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        n_per_cluster=100,
        total_dim=100,
        mean_shift=2.0,
        informative_std=1.0,
        n_repeats=10,
        base_seed=0,
        hdbscan_min_cluster_size=20,
        hdbscan_min_samples=None,
        hdbscan_metric="euclidean",
        hdbscan_cluster_selection_method="eom",
        hdbscan_allow_single_cluster=False,
    )

    plot_noise_std_sweep(noise_summary, metric="ari")
    plot_noise_std_sweep(noise_summary, metric="clustered_fraction")
    plot_noise_std_sweep(noise_summary, metric="n_clusters")
