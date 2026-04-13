#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gaussian_2d_signal_with_noise_dimensions.py

Generate two Gaussian clusters with informative signal in the first 2 dimensions
and pure noise in the remaining dimensions, where the noise dimensions can have
user-defined standard deviations.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def evaluate_kmeans_raw_and_pca(
    X: np.ndarray,
    y: np.ndarray,
    pca_components: int = 2,
    random_state: int = 0,
) -> dict:
    raw_kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    raw_labels = raw_kmeans.fit_predict(X)
    raw_ari = adjusted_rand_score(y, raw_labels)
    raw_sep = between_within_distance_ratio(X, y)

    n_components = min(pca_components, X.shape[1])
    X_pca = PCA(n_components=n_components).fit_transform(X)

    pca_kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    pca_labels = pca_kmeans.fit_predict(X_pca)
    pca_ari = adjusted_rand_score(y, pca_labels)
    pca_sep = between_within_distance_ratio(X_pca, y)

    return {
        "raw_ari": float(raw_ari),
        "pca_ari": float(pca_ari),
        "raw_sep_ratio": float(raw_sep),
        "pca_sep_ratio": float(pca_sep),
    }


def sweep_total_dimension(
    total_dims: Sequence[int],
    n_per_cluster: int = 100,
    mean_shift: float = 2.0,
    informative_std: float = 1.0,
    noise_stds: float | Sequence[float] = 1.0,
    n_repeats: int = 10,
    pca_components: int = 2,
    base_seed: int = 0,
) -> pd.DataFrame:
    rows = []

    for total_dim in total_dims:
        raw_ari_values = []
        pca_ari_values = []
        raw_sep_values = []
        pca_sep_values = []

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

            metrics = evaluate_kmeans_raw_and_pca(
                X=X,
                y=y,
                pca_components=pca_components,
                random_state=seed,
            )

            raw_ari_values.append(metrics["raw_ari"])
            pca_ari_values.append(metrics["pca_ari"])
            raw_sep_values.append(metrics["raw_sep_ratio"])
            pca_sep_values.append(metrics["pca_sep_ratio"])

        row = {
            "total_dim": int(total_dim),
            "raw_ari_mean": float(np.mean(raw_ari_values)),
            "raw_ari_sd": float(np.std(raw_ari_values, ddof=1)),
            "pca_ari_mean": float(np.mean(pca_ari_values)),
            "pca_ari_sd": float(np.std(pca_ari_values, ddof=1)),
            "raw_sep_ratio_mean": float(np.mean(raw_sep_values)),
            "raw_sep_ratio_sd": float(np.std(raw_sep_values, ddof=1)),
            "pca_sep_ratio_mean": float(np.mean(pca_sep_values)),
            "pca_sep_ratio_sd": float(np.std(pca_sep_values, ddof=1)),
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
    pca_components: int = 2,
    base_seed: int = 0,
) -> pd.DataFrame:
    rows = []

    for noise_std in noise_std_values:
        raw_ari_values = []
        pca_ari_values = []
        raw_sep_values = []
        pca_sep_values = []

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

            metrics = evaluate_kmeans_raw_and_pca(
                X=X,
                y=y,
                pca_components=pca_components,
                random_state=seed,
            )

            raw_ari_values.append(metrics["raw_ari"])
            pca_ari_values.append(metrics["pca_ari"])
            raw_sep_values.append(metrics["raw_sep_ratio"])
            pca_sep_values.append(metrics["pca_sep_ratio"])

        row = {
            "noise_std": float(noise_std),
            "total_dim": int(total_dim),
            "raw_ari_mean": float(np.mean(raw_ari_values)),
            "raw_ari_sd": float(np.std(raw_ari_values, ddof=1)),
            "pca_ari_mean": float(np.mean(pca_ari_values)),
            "pca_ari_sd": float(np.std(pca_ari_values, ddof=1)),
            "raw_sep_ratio_mean": float(np.mean(raw_sep_values)),
            "raw_sep_ratio_sd": float(np.std(raw_sep_values, ddof=1)),
            "pca_sep_ratio_mean": float(np.mean(pca_sep_values)),
            "pca_sep_ratio_sd": float(np.std(pca_sep_values, ddof=1)),
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
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=point_size, alpha=alpha, label="Cluster 1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=point_size, alpha=alpha, label="Cluster 2")

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


def plot_pca_2d_view(
    X: np.ndarray,
    y: np.ndarray,
    total_dim: int | None = None,
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
    X_pca = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s=point_size, alpha=alpha, label="Cluster 1")
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s=point_size, alpha=alpha, label="Cluster 2")

    plt.xlabel("PC1", fontsize=xlabel_fontsize)
    plt.ylabel("PC2", fontsize=ylabel_fontsize)

    if title is None:
        if total_dim is not None:
            title = f"2D PCA projection of full data (total_dim={total_dim})"
        else:
            title = "2D PCA projection of full data"

    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if show:
        plt.show()

    return X_pca


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
        plt.plot(x, summary_df["raw_ari_mean"], marker="o", linewidth=2, label="KMeans on raw data")
        plt.plot(x, summary_df["pca_ari_mean"], marker="s", linewidth=2, label="KMeans after PCA to 2D")
        ylabel = "Adjusted Rand Index"
        if title is None:
            title = "Clustering performance vs total dimension"
    elif metric == "sep_ratio":
        plt.plot(
            x,
            summary_df["raw_sep_ratio_mean"],
            marker="o",
            linewidth=2,
            label="Raw between/within distance ratio",
        )
        plt.plot(
            x,
            summary_df["pca_sep_ratio_mean"],
            marker="s",
            linewidth=2,
            label="PCA-2D between/within ratio",
        )
        ylabel = "Mean between / within distance"
        if title is None:
            title = "Cluster separation vs total dimension"
    else:
        raise ValueError('metric must be "ari" or "sep_ratio"')

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
        plt.plot(x, summary_df["raw_ari_mean"], marker="o", linewidth=2, label="KMeans on raw data")
        plt.plot(x, summary_df["pca_ari_mean"], marker="s", linewidth=2, label="KMeans after PCA to 2D")
        ylabel = "Adjusted Rand Index"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"Clustering performance vs noise SD (total_dim={total_dim})"
    elif metric == "sep_ratio":
        plt.plot(
            x,
            summary_df["raw_sep_ratio_mean"],
            marker="o",
            linewidth=2,
            label="Raw between/within distance ratio",
        )
        plt.plot(
            x,
            summary_df["pca_sep_ratio_mean"],
            marker="s",
            linewidth=2,
            label="PCA-2D between/within ratio",
        )
        ylabel = "Mean between / within distance"
        if title is None:
            total_dim = int(summary_df["total_dim"].iloc[0])
            title = f"Cluster separation vs noise SD (total_dim={total_dim})"
    else:
        raise ValueError('metric must be "ari" or "sep_ratio"')

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

    metrics = evaluate_kmeans_raw_and_pca(
        X=X,
        y=y,
        pca_components=2,
        random_state=0,
    )

    print("\\nSingle dataset metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    plot_informative_2d_view(
        X,
        y,
        mean_shift=2.0,
        informative_std=1.0,
    )

    plot_pca_2d_view(
        X,
        y,
        total_dim=total_dim,
    )

    noise_summary = sweep_noise_std(
        noise_std_values=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        n_per_cluster=100,
        total_dim=100,
        mean_shift=2.0,
        informative_std=1.0,
        n_repeats=10,
        pca_components=2,
        base_seed=0,
    )

    plot_noise_std_sweep(noise_summary, metric="ari")
    plot_noise_std_sweep(noise_summary, metric="sep_ratio")
