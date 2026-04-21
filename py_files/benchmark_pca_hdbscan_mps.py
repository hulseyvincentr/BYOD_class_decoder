#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _safe_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _parse_pc_list(text: str) -> List[int]:
    pcs = []
    for token in text.split(','):
        token = token.strip()
        if token:
            pcs.append(int(token))
    if not pcs:
        raise ValueError('pc-list must contain at least one integer')
    return pcs


def _make_hdbscan(min_cluster_size: int, min_samples: Optional[int], cluster_selection_epsilon: float, cluster_selection_method: str):
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        return SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=True,
            copy=False,
        )
    except Exception:
        import hdbscan
        return hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            prediction_data=False,
            gen_min_span_tree=False,
            core_dist_n_jobs=1,
            allow_single_cluster=True,
        )


def _summarize_labels(labels: np.ndarray) -> Tuple[int, int, float]:
    unique, counts = np.unique(labels, return_counts=True)
    counts_map = {int(k): int(v) for k, v in zip(unique, counts)}
    n_clusters = sum(1 for k in counts_map if k != -1)
    n_noise = counts_map.get(-1, 0)
    noise_fraction = float(n_noise) / float(len(labels)) if len(labels) else np.nan
    return n_clusters, n_noise, noise_fraction


def _sklearn_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, float]:
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return X_pca.astype(np.float32, copy=False), explained


def _torch_cov_eigh_pca(X: np.ndarray, n_components: int, device: str) -> Tuple[np.ndarray, float]:
    import torch

    X_np = np.asarray(X, dtype=np.float32, order='C')
    X_t = torch.from_numpy(X_np).to(device)
    X_centered = X_t - X_t.mean(dim=0, keepdim=True)
    n = X_centered.shape[0]
    cov = (X_centered.T @ X_centered) / max(1, n - 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    k = int(n_components)
    comps = evecs[:, :k]
    scores = X_centered @ comps
    total_var = torch.clamp(evals.sum(), min=1e-12)
    explained = float((evals[:k].sum() / total_var).detach().cpu().numpy())
    X_pca = scores.detach().cpu().numpy().astype(np.float32, copy=False)
    return X_pca, explained


def _resolve_mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def benchmark_npz(
    npz_path: str | Path,
    out_dir: str | Path,
    array_key: str = 'predictions',
    pc_list: List[int] | None = None,
    scale_before_pca: bool = False,
    benchmark_torch_cpu: bool = True,
    benchmark_torch_mps: bool = True,
    benchmark_hdbscan: bool = True,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
) -> Dict[str, str]:
    if pc_list is None:
        pc_list = [10, 20, 30]

    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    if array_key not in data:
        raise KeyError(f'{array_key!r} not found in {npz_path}')
    X = _safe_2d(data[array_key]).astype(np.float32)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]

    if scale_before_pca:
        X = StandardScaler(copy=False).fit_transform(X).astype(np.float32, copy=False)

    valid_pc_list = [int(k) for k in pc_list if 1 <= int(k) <= min(X.shape)]
    if not valid_pc_list:
        raise ValueError(f'No valid PCA dimensions in {pc_list} for data shape {X.shape}')

    mps_available = _resolve_mps_available()
    rows = []
    pca_cache: Dict[Tuple[str, int], np.ndarray] = {}

    for n_pc in valid_pc_list:
        t0 = time.perf_counter()
        X_pca, explained = _sklearn_pca(X, n_pc)
        elapsed = time.perf_counter() - t0
        pca_cache[('sklearn_cpu', n_pc)] = X_pca
        rows.append({'backend': 'sklearn_cpu', 'stage': 'pca', 'n_components': n_pc, 'elapsed_seconds': elapsed, 'explained_variance_ratio_sum': explained, 'status': 'ok', 'note': ''})

        if benchmark_torch_cpu:
            t0 = time.perf_counter()
            try:
                X_pca_t, explained_t = _torch_cov_eigh_pca(X, n_pc, device='cpu')
                elapsed_t = time.perf_counter() - t0
                pca_cache[('torch_cpu_cov_eigh', n_pc)] = X_pca_t
                rows.append({'backend': 'torch_cpu_cov_eigh', 'stage': 'pca', 'n_components': n_pc, 'elapsed_seconds': elapsed_t, 'explained_variance_ratio_sum': explained_t, 'status': 'ok', 'note': ''})
            except Exception as e:
                rows.append({'backend': 'torch_cpu_cov_eigh', 'stage': 'pca', 'n_components': n_pc, 'elapsed_seconds': np.nan, 'explained_variance_ratio_sum': np.nan, 'status': 'error', 'note': f'{type(e).__name__}: {e}'})

        if benchmark_torch_mps:
            t0 = time.perf_counter()
            if not mps_available:
                rows.append({'backend': 'torch_mps_cov_eigh', 'stage': 'pca', 'n_components': n_pc, 'elapsed_seconds': np.nan, 'explained_variance_ratio_sum': np.nan, 'status': 'skipped', 'note': 'torch.backends.mps.is_available() is False'})
            else:
                try:
                    X_pca_mps, explained_mps = _torch_cov_eigh_pca(X, n_pc, device='mps')
                    elapsed_mps = time.perf_counter() - t0
                    pca_cache[('torch_mps_cov_eigh', n_pc)] = X_pca_mps
                    rows.append({'backend': 'torch_mps_cov_eigh', 'stage': 'pca', 'n_components': n_pc, 'elapsed_seconds': elapsed_mps, 'explained_variance_ratio_sum': explained_mps, 'status': 'ok', 'note': ''})
                except Exception as e:
                    rows.append({'backend': 'torch_mps_cov_eigh', 'stage': 'pca', 'n_components': n_pc, 'elapsed_seconds': np.nan, 'explained_variance_ratio_sum': np.nan, 'status': 'error', 'note': f'{type(e).__name__}: {e}'})

    if benchmark_hdbscan:
        for n_pc in valid_pc_list:
            clusterer = _make_hdbscan(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method=cluster_selection_method)
            t0 = time.perf_counter()
            labels = clusterer.fit_predict(pca_cache[('sklearn_cpu', n_pc)]).astype(int)
            elapsed = time.perf_counter() - t0
            n_clusters, n_noise, noise_fraction = _summarize_labels(labels)
            rows.append({'backend': 'sklearn_cpu', 'stage': 'hdbscan_after_sklearn_pca', 'n_components': n_pc, 'elapsed_seconds': elapsed, 'explained_variance_ratio_sum': np.nan, 'status': 'ok', 'note': f'clusters={n_clusters}; n_noise={n_noise}; noise_fraction={noise_fraction:.6f}'})

    df = pd.DataFrame(rows)
    stem = npz_path.stem
    csv_path = out_dir / f'{stem}_pca_hdbscan_benchmark.csv'
    df.to_csv(csv_path, index=False)

    summary = {
        'npz_path': str(npz_path),
        'n_points': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'mps_available': bool(mps_available),
        'pc_list': valid_pc_list,
        'benchmark_csv': str(csv_path),
    }
    json_path = out_dir / f'{stem}_pca_hdbscan_benchmark_summary.json'
    json_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    return {'benchmark_csv': str(csv_path), 'summary_json': str(json_path)}


def main():
    parser = argparse.ArgumentParser(description='Benchmark PCA backends (sklearn CPU, torch CPU, torch MPS if available) and HDBSCAN timing for one NPZ file.')
    parser.add_argument('--npz-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--array-key', type=str, default='predictions')
    parser.add_argument('--pc-list', type=str, default='10,20,30')
    parser.add_argument('--scale-before-pca', action='store_true')
    parser.add_argument('--no-torch-cpu', action='store_true')
    parser.add_argument('--no-torch-mps', action='store_true')
    parser.add_argument('--no-hdbscan', action='store_true')
    parser.add_argument('--min-cluster-size', type=int, default=50)
    parser.add_argument('--min-samples', type=int, default=None)
    parser.add_argument('--cluster-selection-epsilon', type=float, default=0.0)
    parser.add_argument('--cluster-selection-method', type=str, default='eom', choices=['eom', 'leaf'])
    args = parser.parse_args()

    benchmark_npz(
        npz_path=args.npz_path,
        out_dir=args.out_dir,
        array_key=args.array_key,
        pc_list=_parse_pc_list(args.pc_list),
        scale_before_pca=args.scale_before_pca,
        benchmark_torch_cpu=not args.no_torch_cpu,
        benchmark_torch_mps=not args.no_torch_mps,
        benchmark_hdbscan=not args.no_hdbscan,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        cluster_selection_method=args.cluster_selection_method,
    )


if __name__ == '__main__':
    main()
