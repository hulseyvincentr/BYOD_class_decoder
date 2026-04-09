#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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

    Parameters
    ----------
    n : int
        Number of samples per distribution.
    d : int
        Original dimensionality.
    mean_shift : float
        Amount of separation between the two distributions.
    informative_dims : int
        Number of dimensions in which the two distributions differ in mean.
    cov_scale : float
        Variance scaling for isotropic covariance.
    random_state : int
        Random seed.

    Returns
    -------
    X : ndarray, shape (2n, d)
        Combined data matrix.
    y : ndarray, shape (2n,)
        True labels: 0 for the first distribution, 1 for the second.
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

    Returns
    -------
    results : dict
        Summary metrics.
    X : ndarray
        Original data.
    y : ndarray
        True labels.
    X_pca : ndarray
        PCA-reduced data.
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


if __name__ == "__main__":
    results, X, y, X_pca = evaluate_after_pca(
        n=200,
        d=100,
        p=2,
        mean_shift=2.0,
        informative_dims=10,
        cov_scale=1.0,
        random_state=0,
    )

    print("Single run:")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nSweep:")
    sweep_results = sweep_p_dimensions(
        n=200,
        d=100,
        p_values=[1, 2, 3, 5, 10, 20, 50],
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
