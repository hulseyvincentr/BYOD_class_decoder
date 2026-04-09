#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def generate_two_distributions(
    n=200,
    d=50,
    mean_shift=2.0,
    informative_dims=5,
    cov_scale=1.0,
    random_state=0,
):
    """
    Generate n samples from each of 2 Gaussian distributions in d dimensions.
    """
    if informative_dims > d:
        raise ValueError("informative_dims must be <= d")

    rng = np.random.default_rng(random_state)

    mean1 = np.zeros(d)
    mean2 = np.zeros(d)
    mean2[:informative_dims] = mean_shift

    cov = np.eye(d) * cov_scale

    X1 = rng.multivariate_normal(mean1, cov, size=n)
    X2 = rng.multivariate_normal(mean2, cov, size=n)

    X = np.vstack([X1, X2])
    y = np.array([0] * n + [1] * n)

    return X, y


def centroid_distance(X, y):
    """
    Euclidean distance between the centroids of the two groups.
    """
    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)
    return np.linalg.norm(c1 - c0)


def pooled_mahalanobis_distance(X, y):
    """
    Mahalanobis distance between the two group centroids using pooled covariance.
    """
    X0 = X[y == 0]
    X1 = X[y == 1]

    c0 = X0.mean(axis=0)
    c1 = X1.mean(axis=0)

    S0 = np.atleast_2d(np.cov(X0, rowvar=False))
    S1 = np.atleast_2d(np.cov(X1, rowvar=False))
    Sp = 0.5 * (S0 + S1)

    Sp_inv = np.linalg.pinv(Sp)
    diff = np.atleast_1d(c1 - c0)
    return float(np.sqrt(diff.T @ Sp_inv @ diff))


def evaluate_after_pca(
    n=200,
    d=50,
    p=2,
    mean_shift=2.0,
    informative_dims=5,
    cov_scale=1.0,
    random_state=0,
):
    """
    Generate two distributions in d dimensions, reduce to p dimensions with PCA,
    and compute distances and simple clustering metrics.
    """
    if p > d:
        raise ValueError("p must be <= d")
    if p < 1:
        raise ValueError("p must be >= 1")

    X, y = generate_two_distributions(
        n=n,
        d=d,
        mean_shift=mean_shift,
        informative_dims=informative_dims,
        cov_scale=cov_scale,
        random_state=random_state,
    )

    raw_centroid_dist = centroid_distance(X, y)

    pca = PCA(n_components=p)
    X_pca = pca.fit_transform(X)

    pca_centroid_dist = centroid_distance(X_pca, y)

    raw_mahal = pooled_mahalanobis_distance(X, y)
    pca_mahal = pooled_mahalanobis_distance(X_pca, y)

    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_pca)

    ari = adjusted_rand_score(y, cluster_labels)

    if p >= 2:
        sil = silhouette_score(X_pca, cluster_labels)
    else:
        sil = float("nan")

    results = {
        "n_per_distribution": int(n),
        "original_dimensions_d": int(d),
        "pca_dimensions_p": int(p),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "raw_centroid_distance": float(raw_centroid_dist),
        "pca_centroid_distance": float(pca_centroid_dist),
        "raw_mahalanobis_distance": float(raw_mahal),
        "pca_mahalanobis_distance": float(pca_mahal),
        "kmeans_ari_after_pca": float(ari),
        "kmeans_silhouette_after_pca": float(sil),
    }

    return results, X, y, X_pca


def sweep_p_dimensions(
    n=200,
    d=50,
    p_values=(1, 2, 3, 5, 10, 20),
    mean_shift=2.0,
    informative_dims=5,
    cov_scale=1.0,
    random_state=0,
):
    """
    Repeat the PCA-distance evaluation across multiple p values.
    """
    all_results = []

    for p in p_values:
        results, _, _, _ = evaluate_after_pca(
            n=n,
            d=d,
            p=p,
            mean_shift=mean_shift,
            informative_dims=informative_dims,
            cov_scale=cov_scale,
            random_state=random_state,
        )
        all_results.append(results)

    return all_results


def plot_distance_vs_p(
    sweep_results,
    metric="pca_centroid_distance",
    title=None,
    marker="o",
    linewidth=2,
    xlabel_fontsize=14,
    ylabel_fontsize=14,
    tick_fontsize=12,
    title_fontsize=15,
    show=True,
):
    """
    Plot how a chosen distance metric changes across PCA dimensions.

    Parameters
    ----------
    sweep_results : list of dict
        Output of sweep_p_dimensions.
    metric : str
        One of:
            "pca_centroid_distance"
            "pca_mahalanobis_distance"
            "explained_variance_ratio_sum"
            "kmeans_ari_after_pca"
            "kmeans_silhouette_after_pca"
    """
    if len(sweep_results) == 0:
        raise ValueError("sweep_results is empty")

    valid_metrics = {
        "pca_centroid_distance": "Centroid distance after PCA",
        "pca_mahalanobis_distance": "Mahalanobis distance after PCA",
        "explained_variance_ratio_sum": "Explained variance ratio sum",
        "kmeans_ari_after_pca": "KMeans ARI after PCA",
        "kmeans_silhouette_after_pca": "KMeans silhouette after PCA",
    }

    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {list(valid_metrics.keys())}")

    p_values = [r["pca_dimensions_p"] for r in sweep_results]
    metric_values = [r[metric] for r in sweep_results]

    plt.figure(figsize=(7, 5))
    plt.plot(p_values, metric_values, marker=marker, linewidth=linewidth)

    plt.xlabel("Number of principal components (p)", fontsize=xlabel_fontsize)
    plt.ylabel(valid_metrics[metric], fontsize=ylabel_fontsize)

    if title is None:
        title = f"{valid_metrics[metric]} vs PCA dimensions"
    plt.title(title, fontsize=title_fontsize)

    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()

    if show:
        plt.show()


def plot_two_distances_vs_p(
    sweep_results,
    xlabel_fontsize=14,
    ylabel_fontsize=14,
    tick_fontsize=12,
    title_fontsize=15,
    legend_fontsize=12,
    show=True,
):
    """
    Plot centroid distance and Mahalanobis distance together across PCA dimensions.
    """
    if len(sweep_results) == 0:
        raise ValueError("sweep_results is empty")

    p_values = [r["pca_dimensions_p"] for r in sweep_results]
    centroid_values = [r["pca_centroid_distance"] for r in sweep_results]
    mahal_values = [r["pca_mahalanobis_distance"] for r in sweep_results]

    plt.figure(figsize=(7, 5))
    plt.plot(p_values, centroid_values, marker="o", linewidth=2, label="Centroid distance")
    plt.plot(p_values, mahal_values, marker="s", linewidth=2, label="Mahalanobis distance")

    plt.xlabel("Number of principal components (p)", fontsize=xlabel_fontsize)
    plt.ylabel("Distance", fontsize=ylabel_fontsize)
    plt.title("Distance vs PCA dimensions", fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if show:
        plt.show()


if __name__ == "__main__":
    p_values = [1, 2, 3, 5, 10, 20, 50]

    sweep_results = sweep_p_dimensions(
        n=200,
        d=100,
        p_values=p_values,
        mean_shift=2.0,
        informative_dims=10,
        cov_scale=1.0,
        random_state=0,
    )

    for r in sweep_results:
        sil_text = "nan" if np.isnan(r["kmeans_silhouette_after_pca"]) else f'{r["kmeans_silhouette_after_pca"]:.3f}'
        print(
            f'p={r["pca_dimensions_p"]}, '
            f'centroid_dist={r["pca_centroid_distance"]:.3f}, '
            f'mahal={r["pca_mahalanobis_distance"]:.3f}, '
            f'ARI={r["kmeans_ari_after_pca"]:.3f}, '
            f'silhouette={sil_text}'
        )

    plot_distance_vs_p(sweep_results, metric="pca_centroid_distance")
