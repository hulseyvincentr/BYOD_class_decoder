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

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _safe_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _parse_pc_list(text: str) -> List[int]:
    pcs = []
    for token in text.split(","):
        token = token.strip()
        if token:
            pcs.append(int(token))
    if not pcs:
        raise ValueError("pc-list must contain at least one integer")
    return pcs


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


def _fallback_color(position: int):
    cmap = plt.get_cmap("tab20")
    return cmap(position % cmap.N)


def _label_to_color(label: int, palette: Dict[int, str], position: int):
    if label in palette:
        return palette[label]
    return _fallback_color(position)


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
            copy=False,
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


def _summarize_labels(labels: np.ndarray) -> Tuple[int, int, float, Dict[int, int]]:
    unique, counts = np.unique(labels, return_counts=True)
    counts_map = {int(k): int(v) for k, v in zip(unique, counts)}
    n_clusters = sum(1 for k in counts_map if k != -1)
    n_noise = counts_map.get(-1, 0)
    noise_fraction = float(n_noise) / float(len(labels)) if len(labels) else np.nan
    return n_clusters, n_noise, noise_fraction, counts_map


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
            title="Label",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_transition_heatmap(T: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("To cluster")
    ax.set_ylabel("From cluster")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def boundary_mask_from_indices(file_indices: np.ndarray) -> np.ndarray:
    return np.diff(file_indices) != 0


def transition_to_probability(T: np.ndarray) -> np.ndarray:
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def _normalize_nonnoise_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    labels = np.asarray(labels).astype(int)
    nonnoise = sorted(int(x) for x in np.unique(labels) if x >= 0)
    old_to_new = {old: new for new, old in enumerate(nonnoise)}
    normalized = labels.copy()
    for old, new in old_to_new.items():
        normalized[labels == old] = new
    return normalized, old_to_new


def build_transition_matrix_ignore_noise(
    labels: np.ndarray,
    boundary_mask: Optional[np.ndarray] = None,
    seq_len: Optional[int] = None,
) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    normalized, _ = _normalize_nonnoise_labels(labels)

    valid = (labels[:-1] >= 0) & (labels[1:] >= 0)
    if boundary_mask is not None:
        valid &= ~boundary_mask
    elif seq_len is not None:
        idx = np.arange(len(labels) - 1)
        valid &= ((idx + 1) % int(seq_len) != 0)

    nonnoise = normalized[normalized >= 0]
    if nonnoise.size == 0:
        raise ValueError("No non-noise labels available for transition merging.")

    k = int(nonnoise.max()) + 1
    T = np.zeros((k, k), dtype=np.float64)

    src = normalized[:-1][valid]
    dst = normalized[1:][valid]
    np.add.at(T, (src, dst), 1)
    return T


def merge_by_transitions(T: np.ndarray, n_merge: int, method: str = "average") -> np.ndarray:
    if T.shape[0] <= 1:
        return np.zeros((T.shape[0],), dtype=int)

    P = transition_to_probability(T)
    S = (P + P.T) / 2.0
    np.fill_diagonal(S, 0)

    D = 1.0 - S
    np.fill_diagonal(D, 0)
    D = np.clip(D, 0, None)

    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method=method)
    group_labels = fcluster(Z, t=n_merge, criterion="maxclust")

    unique = np.unique(group_labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[g] for g in group_labels], dtype=int)


def apply_merge(labels: np.ndarray, merge_map: np.ndarray) -> np.ndarray:
    return merge_map[np.asarray(labels).astype(int)]


def merge_hdbscan_labels(
    labels: np.ndarray,
    file_indices: Optional[np.ndarray] = None,
    n_merge: int = 10,
    method: str = "average",
    seq_len: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray | int]]:
    labels = np.asarray(labels).astype(int)
    nonnoise_unique = sorted(int(x) for x in np.unique(labels) if x >= 0)
    k_before = len(nonnoise_unique)

    if k_before == 0:
        raise ValueError("No non-noise HDBSCAN labels available for merging.")
    if n_merge < 1:
        raise ValueError("n_merge must be >= 1")

    normalized, _ = _normalize_nonnoise_labels(labels)
    boundary_mask = None
    if file_indices is not None:
        boundary_mask = boundary_mask_from_indices(np.asarray(file_indices))

    T_before = build_transition_matrix_ignore_noise(
        normalized,
        boundary_mask=boundary_mask,
        seq_len=seq_len,
    )

    target = min(int(n_merge), int(T_before.shape[0]))
    merge_map = merge_by_transitions(T_before, n_merge=target, method=method)

    merged_labels = labels.copy()
    nonnoise_mask = normalized >= 0
    merged_labels[nonnoise_mask] = apply_merge(normalized[nonnoise_mask], merge_map)

    T_after = build_transition_matrix_ignore_noise(
        merged_labels,
        boundary_mask=boundary_mask,
        seq_len=seq_len,
    )

    info = {
        "T_before": T_before,
        "T_after": T_after,
        "merge_map": merge_map,
        "k_before": int(T_before.shape[0]),
        "k_after": int(T_after.shape[0]),
        "used_boundary_mask": int(boundary_mask is not None),
    }
    return merged_labels, info


def run_pca_hdbscan_merge_sweep(
    npz_path: str | Path,
    out_dir: str | Path,
    array_key: str = "predictions",
    plot_key: str = "embedding_outputs",
    file_index_key: str = "file_indices",
    pc_list: Sequence[int] = (10, 20, 30, 50),
    scale_before_pca: bool = False,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    palette_json: Optional[str] = None,
    point_size: float = 2.0,
    alpha: float = 0.85,
    n_merge: Optional[int] = None,
    merge_method: str = "average",
    use_boundary_mask: bool = True,
    merge_seq_len: Optional[int] = None,
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

    X = _safe_2d(data[array_key]).astype(np.float32)
    plot_xy = _safe_2d(data[plot_key]).astype(np.float32)

    if plot_xy.shape[1] < 2:
        raise ValueError(f"{plot_key!r} must have at least 2 columns for plotting")
    plot_xy = plot_xy[:, :2]

    finite_mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(plot_xy), axis=1)
    if not np.any(finite_mask):
        raise ValueError("No finite rows remain after filtering")

    X = X[finite_mask]
    plot_xy = plot_xy[finite_mask]
    original_indices = np.flatnonzero(finite_mask)

    file_indices = None
    if file_index_key in data:
        file_indices = np.asarray(data[file_index_key])[finite_mask]

    if scale_before_pca:
        X = StandardScaler(copy=False).fit_transform(X).astype(np.float32, copy=False)

    palette = _load_palette(palette_json)

    valid_pc_list = []
    for n_pc in pc_list:
        if 1 <= int(n_pc) <= min(X.shape[0], X.shape[1]):
            valid_pc_list.append(int(n_pc))
    if not valid_pc_list:
        raise ValueError(
            f"No valid n_components values. Data shape is {X.shape}; requested {list(pc_list)}"
        )

    stem = npz_path.stem
    result_dir = out_dir / f"{stem}_pca_hdbscan_merge"
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    raw_panel_records = []
    merged_panel_records = []

    for n_pc in valid_pc_list:
        pca = PCA(n_components=n_pc, random_state=0)
        X_pca = pca.fit_transform(X).astype(np.float32, copy=False)
        explained = float(np.sum(pca.explained_variance_ratio_))

        clusterer = _make_hdbscan(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(X_pca).astype(int)

        n_clusters, n_noise, noise_fraction, counts_map = _summarize_labels(labels)

        raw_csv = result_dir / f"{stem}_pca{n_pc}_hdbscan_labels_raw.csv"
        pd.DataFrame(
            {
                "original_row_index": original_indices,
                "label": labels,
            }
        ).to_csv(raw_csv, index=False)

        if save_labels_npz:
            raw_npz = result_dir / f"{stem}_pca{n_pc}_hdbscan_labels_raw.npz"
            np.savez_compressed(
                raw_npz,
                original_row_index=original_indices,
                labels=labels,
                n_components=np.array([n_pc]),
                explained_variance_ratio_sum=np.array([explained]),
            )

        raw_title = (
            f"{stem} | PCA {n_pc} -> HDBSCAN raw\n"
            f"clusters={n_clusters}, noise={noise_fraction:.2%}, var={explained:.2%}"
        )
        raw_plot = result_dir / f"{stem}_pca{n_pc}_hdbscan_raw_on_{plot_key}.png"
        _save_individual_plot(
            save_path=raw_plot,
            plot_xy=plot_xy,
            labels=labels,
            palette=palette,
            point_size=point_size,
            alpha=alpha,
            title=raw_title,
        )
        raw_panel_records.append({"title": raw_title, "labels": labels})

        merged_labels = None
        k_before_merge = n_clusters
        k_after_merge = n_clusters
        merge_applied = False
        merged_csv = ""
        merged_plot = ""
        T_before_path = ""
        T_after_path = ""
        merge_map_path = ""

        if n_merge is not None and n_clusters > 0:
            target_merge = min(int(n_merge), int(n_clusters))
            if target_merge < n_clusters:
                merged_labels, info = merge_hdbscan_labels(
                    labels=labels,
                    file_indices=(file_indices if use_boundary_mask else None),
                    n_merge=target_merge,
                    method=merge_method,
                    seq_len=merge_seq_len,
                )
                k_before_merge = int(info["k_before"])
                k_after_merge = int(info["k_after"])
                merge_applied = True

                merged_csv_path = result_dir / f"{stem}_pca{n_pc}_hdbscan_labels_merged.csv"
                pd.DataFrame(
                    {
                        "original_row_index": original_indices,
                        "label": merged_labels,
                    }
                ).to_csv(merged_csv_path, index=False)
                merged_csv = str(merged_csv_path)

                if save_labels_npz:
                    merged_npz = result_dir / f"{stem}_pca{n_pc}_hdbscan_labels_merged.npz"
                    np.savez_compressed(
                        merged_npz,
                        original_row_index=original_indices,
                        labels=merged_labels,
                        n_components=np.array([n_pc]),
                        explained_variance_ratio_sum=np.array([explained]),
                    )

                T_before_path_obj = result_dir / f"{stem}_pca{n_pc}_merge_transition_matrix_before.npy"
                T_after_path_obj = result_dir / f"{stem}_pca{n_pc}_merge_transition_matrix_after.npy"
                merge_map_path_obj = result_dir / f"{stem}_pca{n_pc}_merge_map.npy"
                np.save(T_before_path_obj, info["T_before"])
                np.save(T_after_path_obj, info["T_after"])
                np.save(merge_map_path_obj, info["merge_map"])
                T_before_path = str(T_before_path_obj)
                T_after_path = str(T_after_path_obj)
                merge_map_path = str(merge_map_path_obj)

                before_heatmap = result_dir / f"{stem}_pca{n_pc}_merge_transition_matrix_before.png"
                after_heatmap = result_dir / f"{stem}_pca{n_pc}_merge_transition_matrix_after.png"
                _save_transition_heatmap(info["T_before"], f"{stem} PCA {n_pc} transitions before merge", before_heatmap)
                _save_transition_heatmap(info["T_after"], f"{stem} PCA {n_pc} transitions after merge", after_heatmap)

                merged_title = (
                    f"{stem} | PCA {n_pc} -> HDBSCAN merged\n"
                    f"k_before={k_before_merge}, k_after={k_after_merge}, noise={noise_fraction:.2%}"
                )
                merged_plot_path = result_dir / f"{stem}_pca{n_pc}_hdbscan_merged_on_{plot_key}.png"
                _save_individual_plot(
                    save_path=merged_plot_path,
                    plot_xy=plot_xy,
                    labels=merged_labels,
                    palette=palette,
                    point_size=point_size,
                    alpha=alpha,
                    title=merged_title,
                )
                merged_plot = str(merged_plot_path)
                merged_panel_records.append({"title": merged_title, "labels": merged_labels})
            else:
                merged_panel_records.append({"title": raw_title + "\n(merge skipped)", "labels": labels})

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
                "n_clusters_excluding_noise_raw": int(n_clusters),
                "n_noise_raw": int(n_noise),
                "noise_fraction_raw": float(noise_fraction),
                "cluster_sizes_json_raw": json.dumps(counts_map, sort_keys=True),
                "merge_applied": bool(merge_applied),
                "n_merge_target": "" if n_merge is None else int(min(int(n_merge), max(1, n_clusters))) if n_clusters > 0 else "",
                "k_before_merge": int(k_before_merge),
                "k_after_merge": int(k_after_merge),
                "raw_csv": str(raw_csv),
                "raw_plot": str(raw_plot),
                "merged_csv": merged_csv,
                "merged_plot": merged_plot,
                "T_before_npy": T_before_path,
                "T_after_npy": T_after_path,
                "merge_map_npy": merge_map_path,
            }
        )

    summary_csv = result_dir / f"{stem}_pca_hdbscan_merge_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    n_panels = len(raw_panel_records)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))

    fig_raw, axes_raw = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 5.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, record in zip(axes_raw.ravel(), raw_panel_records):
        _plot_one_panel(
            ax=ax,
            plot_xy=plot_xy,
            labels=record["labels"],
            palette=palette,
            point_size=point_size,
            alpha=alpha,
            title=record["title"],
        )
    for ax in axes_raw.ravel()[len(raw_panel_records):]:
        ax.axis("off")
    fig_raw.suptitle(f"{stem}: PCA -> HDBSCAN raw labels on {plot_key}", fontsize=16)
    fig_raw.tight_layout(rect=(0, 0, 0.9, 0.96))
    raw_panel_path = result_dir / f"{stem}_pca_hdbscan_raw_panel_on_{plot_key}.png"
    fig_raw.savefig(raw_panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig_raw)

    merged_panel_path = ""
    if merged_panel_records:
        fig_merged, axes_merged = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6 * ncols, 5.5 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for ax, record in zip(axes_merged.ravel(), merged_panel_records):
            _plot_one_panel(
                ax=ax,
                plot_xy=plot_xy,
                labels=record["labels"],
                palette=palette,
                point_size=point_size,
                alpha=alpha,
                title=record["title"],
            )
        for ax in axes_merged.ravel()[len(merged_panel_records):]:
            ax.axis("off")
        fig_merged.suptitle(f"{stem}: PCA -> HDBSCAN merged labels on {plot_key}", fontsize=16)
        fig_merged.tight_layout(rect=(0, 0, 0.9, 0.96))
        merged_panel_path_obj = result_dir / f"{stem}_pca_hdbscan_merged_panel_on_{plot_key}.png"
        fig_merged.savefig(merged_panel_path_obj, dpi=200, bbox_inches="tight")
        plt.close(fig_merged)
        merged_panel_path = str(merged_panel_path_obj)

    result = {
        "result_dir": str(result_dir),
        "summary_csv": str(summary_csv),
        "raw_panel_plot": str(raw_panel_path),
    }
    if merged_panel_path:
        result["merged_panel_plot"] = merged_panel_path

    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run PCA at multiple dimensions on an NPZ array, cluster each PCA space with HDBSCAN, optionally merge non-noise labels using transition-based merging, and plot labels on embedding_outputs."
    )
    parser.add_argument("--npz-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--array-key", type=str, default="predictions")
    parser.add_argument("--plot-key", type=str, default="embedding_outputs")
    parser.add_argument("--file-index-key", type=str, default="file_indices")
    parser.add_argument("--pc-list", type=str, default="10,20,30,50")
    parser.add_argument("--scale-before-pca", action="store_true")
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--cluster-selection-method", type=str, default="eom", choices=["eom", "leaf"])
    parser.add_argument("--palette-json", type=str, default=None)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--no-save-labels-npz", action="store_true")
    parser.add_argument("--n-merge", type=int, default=None)
    parser.add_argument("--merge-method", type=str, default="average")
    parser.add_argument("--merge-seq-len", type=int, default=None)
    parser.add_argument("--no-merge-boundary-mask", action="store_true")

    args = parser.parse_args()

    run_pca_hdbscan_merge_sweep(
        npz_path=args.npz_path,
        out_dir=args.out_dir,
        array_key=args.array_key,
        plot_key=args.plot_key,
        file_index_key=args.file_index_key,
        pc_list=_parse_pc_list(args.pc_list),
        scale_before_pca=args.scale_before_pca,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        cluster_selection_method=args.cluster_selection_method,
        palette_json=args.palette_json,
        point_size=args.point_size,
        alpha=args.alpha,
        n_merge=args.n_merge,
        merge_method=args.merge_method,
        use_boundary_mask=not args.no_merge_boundary_mask,
        merge_seq_len=args.merge_seq_len,
        save_labels_npz=not args.no_save_labels_npz,
    )


if __name__ == "__main__":
    main()
