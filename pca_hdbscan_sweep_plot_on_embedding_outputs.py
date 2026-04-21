#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _safe_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _load_palette(palette_json: Optional[str]) -> Dict[int, str]:
    if not palette_json:
        return {}
    with open(palette_json, "r") as f:
        raw = json.load(f)
    palette: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            palette[int(k)] = str(v)
        except Exception:
            continue
    return palette


def _fallback_color(position: int) -> Tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab20")
    return cmap(position % cmap.N)


def _label_to_color(label: int, palette: Dict[int, str], position: int):
    if label in palette:
        return palette[label]
    return _fallback_color(position)


def _parse_pc_list(text: str) -> List[int]:
    pcs = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        pcs.append(int(token))
    if not pcs:
        raise ValueError("pc-list must contain at least one integer")
    return pcs


def _make_hdbscan(
    min_cluster_size: int,
    min_samples: Optional[int],
    cluster_selection_epsilon: float,
    cluster_selection_method: str,
):
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        return SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=True,
        )
    except Exception:
        try:
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
        except Exception as e:
            raise ImportError(
                "Could not import HDBSCAN from sklearn or the external hdbscan package. "
                "Try `pip install hdbscan` in your active environment."
            ) from e


def _plot_one_panel(
    ax,
    plot_xy: np.ndarray,
    labels: np.ndarray,
    palette: Dict[int, str],
    point_size: float,
    alpha: float,
    title: str,
):
    unique_labels = sorted(int(x) for x in np.unique(labels))
    label_to_position = {lab: i for i, lab in enumerate(unique_labels)}

    for lab in unique_labels:
        mask = labels == lab
        color = _label_to_color(lab, palette, label_to_position[lab])
        label_text = "noise (-1)" if lab == -1 else str(lab)
        ax.scatter(
            plot_xy[mask, 0],
            plot_xy[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[color],
            linewidths=0,
            rasterized=True,
            label=label_text,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _save_individual_plot(
    save_path: Path,
    plot_xy: np.ndarray,
    labels: np.ndarray,
    palette: Dict[int, str],
    point_size: float,
    alpha: float,
    title: str,
):
    fig, ax = plt.subplots(figsize=(8, 7))
    _plot_one_panel(
        ax=ax,
        plot_xy=plot_xy,
        labels=labels,
        palette=palette,
        point_size=point_size,
        alpha=alpha,
        title=title,
    )
    handles, labels_text = ax.get_legend_handles_labels()
    if len(handles) <= 25:
        ax.legend(
            handles,
            labels_text,
            title="HDBSCAN label",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _summarize_labels(labels: np.ndarray) -> Tuple[int, int, float, Dict[int, int]]:
    unique, counts = np.unique(labels, return_counts=True)
    counts_map = {int(k): int(v) for k, v in zip(unique, counts)}
    n_clusters = sum(1 for k in counts_map if k != -1)
    n_noise = counts_map.get(-1, 0)
    noise_fraction = float(n_noise) / float(len(labels)) if len(labels) else np.nan
    return n_clusters, n_noise, noise_fraction, counts_map


def run_pca_hdbscan_sweep(
    npz_path: str | Path,
    out_dir: str | Path,
    array_key: str = "predictions",
    plot_key: str = "embedding_outputs",
    pc_list: Sequence[int] = (2, 5, 10, 20, 30, 50),
    scale_before_pca: bool = False,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    palette_json: Optional[str] = None,
    point_size: float = 2.0,
    alpha: float = 0.85,
    save_labels_npz: bool = True,
) -> Dict[str, str]:
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    if array_key not in data:
        raise KeyError(f"{array_key!r} not found in {npz_path}")
    if plot_key not in data:
        raise KeyError(f"{plot_key!r} not found in {npz_path}")

    X = _safe_2d(data[array_key]).astype(float)
    plot_xy = _safe_2d(data[plot_key]).astype(float)

    if plot_xy.shape[1] < 2:
        raise ValueError(f"{plot_key!r} must have at least 2 columns for plotting")
    plot_xy = plot_xy[:, :2]

    finite_mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(plot_xy), axis=1)
    if not np.any(finite_mask):
        raise ValueError("No finite rows remain after filtering")

    X = X[finite_mask]
    plot_xy = plot_xy[finite_mask]
    original_indices = np.flatnonzero(finite_mask)

    palette = _load_palette(palette_json)

    valid_pc_list = []
    for n_pc in pc_list:
        if n_pc < 1:
            continue
        if n_pc > X.shape[1]:
            continue
        valid_pc_list.append(int(n_pc))
    if not valid_pc_list:
        raise ValueError(
            f"No valid n_components values. Data has {X.shape[1]} columns but requested {list(pc_list)}"
        )

    stem = npz_path.stem
    result_dir = out_dir / f"{stem}_pca_hdbscan"
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    panel_records = []

    for n_pc in valid_pc_list:
        X_in = X
        if scale_before_pca:
            scaler = StandardScaler()
            X_in = scaler.fit_transform(X_in)

        pca = PCA(n_components=n_pc, random_state=0)
        X_pca = pca.fit_transform(X_in)

        clusterer = _make_hdbscan(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(X_pca).astype(int)

        n_clusters, n_noise, noise_fraction, counts_map = _summarize_labels(labels)
        explained = float(np.sum(pca.explained_variance_ratio_))

        summary_rows.append(
            {
                "npz_path": str(npz_path),
                "array_key": array_key,
                "plot_key": plot_key,
                "n_points_used": int(X.shape[0]),
                "n_components": int(n_pc),
                "explained_variance_ratio_sum": explained,
                "min_cluster_size": int(min_cluster_size),
                "min_samples": "" if min_samples is None else int(min_samples),
                "cluster_selection_epsilon": float(cluster_selection_epsilon),
                "cluster_selection_method": cluster_selection_method,
                "n_clusters_excluding_noise": int(n_clusters),
                "n_noise": int(n_noise),
                "noise_fraction": float(noise_fraction),
                "cluster_sizes_json": json.dumps(counts_map, sort_keys=True),
            }
        )

        labels_csv = result_dir / f"{stem}_pca{n_pc}_hdbscan_labels.csv"
        pd.DataFrame(
            {
                "original_row_index": original_indices,
                "label": labels,
            }
        ).to_csv(labels_csv, index=False)

        if save_labels_npz:
            labels_npz = result_dir / f"{stem}_pca{n_pc}_hdbscan_labels.npz"
            np.savez_compressed(
                labels_npz,
                original_row_index=original_indices,
                labels=labels,
                n_components=np.array([n_pc]),
                explained_variance_ratio_sum=np.array([explained]),
            )

        title = (
            f"{stem} | PCA {n_pc} -> HDBSCAN\n"
            f"clusters={n_clusters}, noise={noise_fraction:.2%}, var={explained:.2%}"
        )

        plot_path = result_dir / f"{stem}_pca{n_pc}_hdbscan_on_{plot_key}.png"
        _save_individual_plot(
            save_path=plot_path,
            plot_xy=plot_xy,
            labels=labels,
            palette=palette,
            point_size=point_size,
            alpha=alpha,
            title=title,
        )

        panel_records.append(
            {
                "n_components": n_pc,
                "labels": labels,
                "title": title,
            }
        )

    summary_csv = result_dir / f"{stem}_pca_hdbscan_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    n_panels = len(panel_records)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 5.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax, record in zip(axes.ravel(), panel_records):
        _plot_one_panel(
            ax=ax,
            plot_xy=plot_xy,
            labels=record["labels"],
            palette=palette,
            point_size=point_size,
            alpha=alpha,
            title=record["title"],
        )

    for ax in axes.ravel()[len(panel_records):]:
        ax.axis("off")

    handles, labels_text = axes.ravel()[0].get_legend_handles_labels()
    if len(handles) <= 25:
        fig.legend(
            handles,
            labels_text,
            title="HDBSCAN label",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

    fig.suptitle(f"{stem}: PCA dimension sweep with HDBSCAN plotted on {plot_key}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))
    panel_path = result_dir / f"{stem}_pca_hdbscan_panel_on_{plot_key}.png"
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "result_dir": str(result_dir),
        "summary_csv": str(summary_csv),
        "panel_plot": str(panel_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run PCA at multiple dimensions on an NPZ array, cluster each PCA space with HDBSCAN, and plot labels on embedding_outputs."
    )
    parser.add_argument("--npz-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--array-key", type=str, default="predictions")
    parser.add_argument("--plot-key", type=str, default="embedding_outputs")
    parser.add_argument("--pc-list", type=str, default="2,5,10,20,30,50")
    parser.add_argument("--scale-before-pca", action="store_true")
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--cluster-selection-method", type=str, default="eom", choices=["eom", "leaf"])
    parser.add_argument("--palette-json", type=str, default=None)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--no-save-labels-npz", action="store_true")

    args = parser.parse_args()

    result = run_pca_hdbscan_sweep(
        npz_path=args.npz_path,
        out_dir=args.out_dir,
        array_key=args.array_key,
        plot_key=args.plot_key,
        pc_list=_parse_pc_list(args.pc_list),
        scale_before_pca=args.scale_before_pca,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        cluster_selection_method=args.cluster_selection_method,
        palette_json=args.palette_json,
        point_size=args.point_size,
        alpha=args.alpha,
        save_labels_npz=not args.no_save_labels_npz,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
