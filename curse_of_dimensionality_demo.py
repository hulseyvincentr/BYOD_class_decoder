#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
curse_of_dimensionality_demo.py

A demo script for exploring concentration of distances and its effect
on clustering as dimensionality increases.

What it includes
----------------
1. Concentration of pairwise distances in a single distribution
   - mean pairwise distance vs dimension
   - coefficient of variation (std / mean) vs dimension
   - nearest-vs-farthest contrast vs dimension
   - pairwise distance histograms at selected dimensions

2. Clustering with fixed signal and increasing noise dimensions
   - two Gaussian clusters with separation in only a small number of dimensions
   - KMeans clustering performance (ARI) vs total dimension
   - optional PCA before clustering
   - within/between distance ratio vs total dimension

Dependencies
------------
numpy
matplotlib
scipy
scikit-learn
pandas
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


def generate_single_distribution_points(
    n_points: int = 200,
    d: int = 10,
    distribution: str = "gaussian",
    random_state: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    if distribution == "gaussian":
        X = rng.normal(loc=0.0, scale=1.0, size=(n_points, d))
    elif distribution == "uniform":
        X = rng.uniform(low=0.0, high=1.0, size=(n_points, d))
    else:
        raise ValueError('distribution must be "gaussian" or "uniform"')

    return X


def pairwise_distance_stats(X: np.ndarray) -> dict:
    condensed = pdist(X, metric="euclidean")
    dist_matrix = squareform(condensed)

    np.fill_diagonal(dist_matrix, np.inf)
    nearest = dist_matrix.min(axis=1)

    np.fill_diagonal(dist_matrix, -np.inf)
    farthest = dist_matrix.max(axis=1)

    mean_distance = float(np.mean(condensed))
    std_distance = float(np.std(condensed, ddof=1))
    cv_distance = float(std_distance / mean_distance) if mean_distance > 0 else np.nan
    mean_nearest = float(np.mean(nearest))
    mean_farthest = float(np.mean(farthest))
    contrast = float(np.mean((farthest - nearest) / nearest))

    return {
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "cv_distance": cv_distance,
        "mean_nearest_neighbor_distance": mean_nearest,
        "mean_farthest_neighbor_distance": mean_farthest,
        "nn_farthest_contrast": contrast,
    }


def concentration_summary_over_dimensions(
    dimensions: Sequence[int],
    n_points: int = 200,
    distribution: str = "gaussian",
    n_repeats: int = 10,
    base_seed: int = 0,
) -> pd.DataFrame:
    rows = []

    for d in dimensions:
        per_repeat = []
        for r in range(n_repeats):
            seed = base_seed + 1000 * r + d
            X = generate_single_distribution_points(
                n_points=n_points,
                d=d,
                distribution=distribution,
                random_state=seed,
            )
            stats = pairwise_distance_stats(X)
            per_repeat.append(stats)

        repeat_df = pd.DataFrame(per_repeat)
        mean_row = repeat_df.mean(numeric_only=True).to_dict()
        std_row = repeat_df.std(numeric_only=True, ddof=1).to_dict()

        row = {
            "dimension": d,
            "n_points": n_points,
            "distribution": distribution,
            "n_repeats": n_repeats,
        }

        for key, value in mean_row.items():
            row[key] = float(value)
        for key, value in std_row.items():
            row[f"{key}_across_repeats_sd"] = float(value)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_concentration_metric(
    summary_df: pd.DataFrame,
    metric: str = "cv_distance",
    title: str | None = None,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    tick_fontsize: int = 12,
    title_fontsize: int = 15,
    show: bool = True,
):
    metric_labels = {
        "mean_distance": "Mean pairwise distance",
        "std_distance": "SD of pairwise distances",
        "cv_distance": "Coefficient of variation (SD / mean)",
        "mean_nearest_neighbor_distance": "Mean nearest-neighbor distance",
        "mean_farthest_neighbor_distance": "Mean farthest-neighbor distance",
        "nn_farthest_contrast": "Mean (farthest - nearest) / nearest",
    }

    if metric not in metric_labels:
        raise ValueError(f"Unknown metric: {metric}")

    x = summary_df["dimension"].to_numpy()
    y = summary_df[metric].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Dimension", fontsize=xlabel_fontsize)
    plt.ylabel(metric_labels[metric], fontsize=ylabel_fontsize)

    if title is None:
        distribution = summary_df["distribution"].iloc[0]
        n_points = int(summary_df["n_points"].iloc[0])
        title = f"{metric_labels[metric]} vs dimension ({distribution}, n={n_points})"

    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()

    if show:
        plt.show()


def plot_distance_histograms(
    dimensions: Sequence[int] = (2, 50, 500),
    n_points: int = 200,
    distribution: str = "gaussian",
    base_seed: int = 0,
    bins: int = 40,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    tick_fontsize: int = 12,
    title_fontsize: int = 15,
    show: bool = True,
):
    for d in dimensions:
        X = generate_single_distribution_points(
            n_points=n_points,
            d=d,
            distribution=distribution,
            random_state=base_seed + d,
        )
        distances = pdist(X, metric="euclidean")

        plt.figure(figsize=(7, 5))
        plt.hist(distances, bins=bins)
        plt.xlabel("Pairwise Euclidean distance", fontsize=xlabel_fontsize)
        plt.ylabel("Count", fontsize=ylabel_fontsize)
        plt.title(
            f"Pairwise distance histogram ({distribution}, d={d}, n={n_points})",
            fontsize=title_fontsize,
        )
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()

        if show:
            plt.show()


def generate_two_cluster_data(
    n_per_cluster: int = 100,
    d: int = 20,
    informative_dims: int = 2,
    mean_shift: float = 2.0,
    cov_scale: float = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if informative_dims > d:
        raise ValueError("informative_dims must be <= d")

    rng = np.random.default_rng(random_state)

    mean1 = np.zeros(d)
    mean2 = np.zeros(d)

    half_shift = mean_shift / 2.0
    mean1[:informative_dims] = -half_shift
    mean2[:informative_dims] = half_shift

    cov = np.eye(d) * cov_scale

    X1 = rng.multivariate_normal(mean=mean1, cov=cov, size=n_per_cluster)
    X2 = rng.multivariate_normal(mean=mean2, cov=cov, size=n_per_cluster)

    X = np.vstack([X1, X2])
    y = np.array([0] * n_per_cluster + [1] * n_per_cluster)

    return X, y


def within_between_distance_ratio(X: np.ndarray, y: np.ndarray) -> float:
    dist_matrix = squareform(pdist(X, metric="euclidean"))

    same_mask = y[:, None] == y[None, :]
    diff_mask = ~same_mask

    upper_triangle = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
    within_vals = dist_matrix[same_mask & upper_triangle]
    between_vals = dist_matrix[diff_mask & upper_triangle]

    mean_within = float(np.mean(within_vals))
    mean_between = float(np.mean(between_vals))

    return mean_between / mean_within


def clustering_summary_over_dimensions(
    dimensions: Sequence[int],
    n_per_cluster: int = 100,
    informative_dims: int = 2,
    mean_shift: float = 2.0,
    cov_scale: float = 1.0,
    n_repeats: int = 10,
    base_seed: int = 0,
    pca_components: int | None = 2,
) -> pd.DataFrame:
    rows = []

    for d in dimensions:
        raw_ari_list = []
        pca_ari_list = []
        raw_sep_ratio_list = []
        pca_sep_ratio_list = []

        for r in range(n_repeats):
            seed = base_seed + 1000 * r + d
            X, y = generate_two_cluster_data(
                n_per_cluster=n_per_cluster,
                d=d,
                informative_dims=informative_dims,
                mean_shift=mean_shift,
                cov_scale=cov_scale,
                random_state=seed,
            )

            raw_sep = within_between_distance_ratio(X, y)
            raw_sep_ratio_list.append(raw_sep)

            kmeans_raw = KMeans(n_clusters=2, random_state=seed, n_init=10)
            raw_labels = kmeans_raw.fit_predict(X)
            raw_ari = adjusted_rand_score(y, raw_labels)
            raw_ari_list.append(raw_ari)

            if pca_components is not None:
                n_components = min(pca_components, d)
                X_pca = PCA(n_components=n_components).fit_transform(X)

                pca_sep = within_between_distance_ratio(X_pca, y)
                pca_sep_ratio_list.append(pca_sep)

                kmeans_pca = KMeans(n_clusters=2, random_state=seed, n_init=10)
                pca_labels = kmeans_pca.fit_predict(X_pca)
                pca_ari = adjusted_rand_score(y, pca_labels)
                pca_ari_list.append(pca_ari)

        row = {
            "dimension": d,
            "n_per_cluster": n_per_cluster,
            "informative_dims": informative_dims,
            "mean_shift": mean_shift,
            "cov_scale": cov_scale,
            "n_repeats": n_repeats,
            "raw_ari_mean": float(np.mean(raw_ari_list)),
            "raw_ari_sd": float(np.std(raw_ari_list, ddof=1)),
            "raw_sep_ratio_mean": float(np.mean(raw_sep_ratio_list)),
            "raw_sep_ratio_sd": float(np.std(raw_sep_ratio_list, ddof=1)),
        }

        if pca_components is not None and len(pca_ari_list) > 0:
            row["pca_components"] = int(pca_components)
            row["pca_ari_mean"] = float(np.mean(pca_ari_list))
            row["pca_ari_sd"] = float(np.std(pca_ari_list, ddof=1))
            row["pca_sep_ratio_mean"] = float(np.mean(pca_sep_ratio_list))
            row["pca_sep_ratio_sd"] = float(np.std(pca_sep_ratio_list, ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


def plot_clustering_performance(
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
    x = summary_df["dimension"].to_numpy()

    plt.figure(figsize=(7, 5))

    if metric == "ari":
        plt.plot(x, summary_df["raw_ari_mean"], marker="o", linewidth=2, label="KMeans on raw data")
        if "pca_ari_mean" in summary_df.columns:
            pca_components = int(summary_df["pca_components"].iloc[0])
            plt.plot(
                x,
                summary_df["pca_ari_mean"],
                marker="s",
                linewidth=2,
                label=f"KMeans after PCA to {pca_components}D",
            )
        ylabel = "Adjusted Rand Index"
        if title is None:
            informative_dims = int(summary_df["informative_dims"].iloc[0])
            mean_shift = float(summary_df["mean_shift"].iloc[0])
            title = (
                f"Clustering performance vs dimension "
                f"(informative_dims={informative_dims}, mean_shift={mean_shift})"
            )

    elif metric == "sep_ratio":
        plt.plot(
            x,
            summary_df["raw_sep_ratio_mean"],
            marker="o",
            linewidth=2,
            label="Raw between/within distance ratio",
        )
        if "pca_sep_ratio_mean" in summary_df.columns:
            pca_components = int(summary_df["pca_components"].iloc[0])
            plt.plot(
                x,
                summary_df["pca_sep_ratio_mean"],
                marker="s",
                linewidth=2,
                label=f"PCA-{pca_components}D between/within ratio",
            )
        ylabel = "Mean between / within distance"
        if title is None:
            informative_dims = int(summary_df["informative_dims"].iloc[0])
            mean_shift = float(summary_df["mean_shift"].iloc[0])
            title = (
                f"Cluster separation vs dimension "
                f"(informative_dims={informative_dims}, mean_shift={mean_shift})"
            )
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


def run_all_demos(
    concentration_dimensions: Sequence[int] = (2, 5, 10, 20, 50, 100, 200, 500),
    histogram_dimensions: Sequence[int] = (2, 50, 500),
    clustering_dimensions: Sequence[int] = (2, 5, 10, 20, 50, 100, 200, 500),
    n_points: int = 200,
    n_per_cluster: int = 100,
    informative_dims: int = 2,
    mean_shift: float = 2.0,
    cov_scale: float = 1.0,
    distribution: str = "gaussian",
    n_repeats: int = 10,
    pca_components: int | None = 2,
    base_seed: int = 0,
    out_dir: str | None = None,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    concentration_df = concentration_summary_over_dimensions(
        dimensions=concentration_dimensions,
        n_points=n_points,
        distribution=distribution,
        n_repeats=n_repeats,
        base_seed=base_seed,
    )

    clustering_df = clustering_summary_over_dimensions(
        dimensions=clustering_dimensions,
        n_per_cluster=n_per_cluster,
        informative_dims=informative_dims,
        mean_shift=mean_shift,
        cov_scale=cov_scale,
        n_repeats=n_repeats,
        base_seed=base_seed,
        pca_components=pca_components,
    )

    plot_concentration_metric(concentration_df, metric="cv_distance", show=show_plots)
    plot_concentration_metric(concentration_df, metric="nn_farthest_contrast", show=show_plots)
    plot_distance_histograms(
        dimensions=histogram_dimensions,
        n_points=n_points,
        distribution=distribution,
        base_seed=base_seed,
        show=show_plots,
    )
    plot_clustering_performance(clustering_df, metric="ari", show=show_plots)
    plot_clustering_performance(clustering_df, metric="sep_ratio", show=show_plots)

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        concentration_csv = out_path / "concentration_summary.csv"
        clustering_csv = out_path / "clustering_summary.csv"

        concentration_df.to_csv(concentration_csv, index=False)
        clustering_df.to_csv(clustering_csv, index=False)

        print(f"Saved concentration summary to: {concentration_csv}")
        print(f"Saved clustering summary to: {clustering_csv}")

    return concentration_df, clustering_df


if __name__ == "__main__":
    concentration_df, clustering_df = run_all_demos(
        concentration_dimensions=[2, 5, 10, 20, 50, 100, 200, 500],
        histogram_dimensions=[2, 50, 500],
        clustering_dimensions=[2, 5, 10, 20, 50, 100, 200, 500],
        n_points=200,
        n_per_cluster=100,
        informative_dims=2,
        mean_shift=2.0,
        cov_scale=1.0,
        distribution="gaussian",
        n_repeats=10,
        pca_components=2,
        base_seed=0,
        out_dir=None,
        show_plots=True,
    )

    print("\nConcentration summary:")
    print(concentration_df)

    print("\nClustering summary:")
    print(clustering_df)
    
    
    """
    import sys
import importlib

sys.path.append("/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class_decoder")

import curse_of_dimensionality_demo as cdd
importlib.reload(cdd)

concentration_df, clustering_df = cdd.run_all_demos(
    concentration_dimensions=[2, 5, 10, 20, 50, 100, 200, 500],
    histogram_dimensions=[2, 50, 500],
    clustering_dimensions=[2, 5, 10, 20, 50, 100, 200, 500],
    n_points=200,
    n_per_cluster=100,
    informative_dims=2,
    mean_shift=2.0,
    cov_scale=1.0,
    distribution="gaussian",
    n_repeats=10,
    pca_components=2,
    base_seed=0,
    out_dir=None,
    show_plots=True,
)

print(concentration_df)
print(clustering_df)
    
    """
