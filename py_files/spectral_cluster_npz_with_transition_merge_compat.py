#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    palette = {}
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


def _plot_labels(
    plot_xy: np.ndarray,
    labels: np.ndarray,
    palette: Dict[int, str],
    title: str,
    save_path: Path,
    point_size: float = 2.0,
    alpha: float = 0.85,
) -> None:
    unique_labels = sorted(int(x) for x in np.unique(labels))
    label_to_position = {lab: i for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 7))
    for lab in unique_labels:
        mask = labels == lab
        color = _label_to_color(lab, palette, label_to_position[lab])
        ax.scatter(
            plot_xy[mask, 0],
            plot_xy[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[color],
            linewidths=0,
            rasterized=True,
            label=str(lab),
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    handles, labels_text = ax.get_legend_handles_labels()
    if len(handles) <= 25:
        ax.legend(
            handles,
            labels_text,
            title="Cluster label",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
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


def _choose_spectral_method(method: str, n_points: int) -> str:
    if method != "auto":
        return method
    if n_points < 100_000:
        return "gpu"
    if n_points <= 2_000_000:
        return "nystrom"
    return "twostage"


def _filter_supported_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    supported = set(sig.parameters.keys())
    supported.discard("self")
    return {k: v for k, v in kwargs.items() if k in supported}


def _make_clusterer(
    method: str,
    n_clusters: int,
    n_neighbors: int,
    seed: int,
    n_landmarks: int,
    n_subsample: int,
):
    method = method.lower()
    if method == "gpu":
        from gpu_spectral import GPUSpectral
        kwargs = _filter_supported_kwargs(
            GPUSpectral,
            {
                "n_clusters": n_clusters,
                "n_neighbors": n_neighbors,
                "random_state": seed,
                "seed": seed,
            },
        )
        print(f"[INFO] GPUSpectral kwargs: {kwargs}")
        return GPUSpectral(**kwargs)

    if method == "nystrom":
        from gpu_spectral import NystromSpectral
        kwargs = _filter_supported_kwargs(
            NystromSpectral,
            {
                "n_clusters": n_clusters,
                "n_neighbors": n_neighbors,
                "n_landmarks": n_landmarks,
                "random_state": seed,
                "seed": seed,
            },
        )
        print(f"[INFO] NystromSpectral kwargs: {kwargs}")
        return NystromSpectral(**kwargs)

    if method == "twostage":
        from gpu_spectral import TwoStageSpectral
        kwargs = _filter_supported_kwargs(
            TwoStageSpectral,
            {
                "n_clusters": n_clusters,
                "n_neighbors": n_neighbors,
                "n_subsample": n_subsample,
                "random_state": seed,
                "seed": seed,
            },
        )
        print(f"[INFO] TwoStageSpectral kwargs: {kwargs}")
        return TwoStageSpectral(**kwargs)

    raise ValueError("method must be one of: auto, gpu, nystrom, twostage")


@dataclass
class SpectralMergeConfig:
    array_key: str = "predictions"
    plot_key: str = "embedding_outputs"
    file_index_key: str = "file_indices"
    output_label_key: str = "spectral_labels"
    merged_output_label_key: str = "spectral_labels_merged"
    method: str = "auto"
    n_clusters: int = 20
    n_neighbors: int = 30
    seed: int = 42
    n_landmarks: int = 5000
    n_subsample: int = 10000
    point_size: float = 2.0
    alpha: float = 0.85
    drop_nonfinite_rows: bool = True
    save_augmented_npz: bool = False
    palette_json: Optional[str] = None
    n_merge: Optional[int] = None
    merge_method: str = "average"
    merge_use_boundary_mask: bool = True
    merge_seq_len: Optional[int] = None


def run_spectral_for_npz_with_merge(
    npz_path: str | Path,
    out_dir: str | Path,
    cfg: SpectralMergeConfig,
) -> Dict[str, str]:
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    if cfg.array_key not in data:
        raise KeyError(f"{cfg.array_key!r} not found in {npz_path}")
    if cfg.plot_key not in data:
        raise KeyError(f"{cfg.plot_key!r} not found in {npz_path}")

    X = _safe_2d(data[cfg.array_key]).astype(np.float32)
    plot_xy = _safe_2d(data[cfg.plot_key]).astype(np.float32)
    if plot_xy.shape[1] < 2:
        raise ValueError(f"{cfg.plot_key!r} must have at least 2 columns")
    plot_xy = plot_xy[:, :2]

    n_total = X.shape[0]
    finite_mask = np.ones(n_total, dtype=bool)
    if cfg.drop_nonfinite_rows:
        finite_mask &= np.all(np.isfinite(X), axis=1)
        finite_mask &= np.all(np.isfinite(plot_xy), axis=1)

    if not np.any(finite_mask):
        raise ValueError("No finite rows remain after filtering")

    X = X[finite_mask]
    plot_xy = plot_xy[finite_mask]
    original_indices = np.flatnonzero(finite_mask)

    file_indices = None
    if cfg.file_index_key in data:
        file_indices = np.asarray(data[cfg.file_index_key])[finite_mask]

    palette = _load_palette(cfg.palette_json)

    chosen_method = _choose_spectral_method(cfg.method, X.shape[0])
    clusterer = _make_clusterer(
        method=chosen_method,
        n_clusters=cfg.n_clusters,
        n_neighbors=cfg.n_neighbors,
        seed=cfg.seed,
        n_landmarks=cfg.n_landmarks,
        n_subsample=cfg.n_subsample,
    )

    print(f"[INFO] Running spectral clustering with method={chosen_method}")
    labels = clusterer.fit_predict(X).astype(int)

    stem = npz_path.stem
    result_dir = out_dir / f"{stem}_spectral"
    result_dir.mkdir(parents=True, exist_ok=True)

    raw_plot = result_dir / f"{stem}_spectral_labels_raw.png"
    _plot_labels(
        plot_xy=plot_xy,
        labels=labels,
        palette=palette,
        title=f"{stem} spectral clustering (raw labels)",
        save_path=raw_plot,
        point_size=cfg.point_size,
        alpha=cfg.alpha,
    )

    raw_csv = result_dir / f"{stem}_spectral_labels_raw.csv"
    pd.DataFrame(
        {"original_row_index": original_indices, "label": labels}
    ).to_csv(raw_csv, index=False)

    merged_labels = None
    merge_info_paths = {}
    merge_summary = {
        "merge_applied": False,
        "k_before": int(np.unique(labels).size),
        "k_after": int(np.unique(labels).size),
    }

    if cfg.n_merge is not None:
        k_before = int(np.unique(labels).size)
        if cfg.n_merge < 1:
            raise ValueError("--n-merge must be >= 1")
        if cfg.n_merge >= k_before:
            print(
                f"[INFO] Skipping merge because n_merge={cfg.n_merge} "
                f"is not smaller than k_before={k_before}"
            )
        else:
            from gpu_spectral.merge import merge_clusters, boundary_mask_from_indices

            if cfg.merge_use_boundary_mask and file_indices is not None:
                boundary_mask = boundary_mask_from_indices(file_indices)
                print("[INFO] Using boundary mask from file_indices for merge")
            else:
                boundary_mask = None

            merged_labels, info = merge_clusters(
                labels=labels,
                n_merge=cfg.n_merge,
                seq_len=cfg.merge_seq_len,
                boundary_mask=boundary_mask,
                method=cfg.merge_method,
            )
            merged_labels = merged_labels.astype(int)

            merged_plot = result_dir / f"{stem}_spectral_labels_merged.png"
            _plot_labels(
                plot_xy=plot_xy,
                labels=merged_labels,
                palette=palette,
                title=f"{stem} spectral clustering (merged labels)",
                save_path=merged_plot,
                point_size=cfg.point_size,
                alpha=cfg.alpha,
            )

            merged_csv = result_dir / f"{stem}_spectral_labels_merged.csv"
            pd.DataFrame(
                {"original_row_index": original_indices, "label": merged_labels}
            ).to_csv(merged_csv, index=False)

            T_before_path = result_dir / f"{stem}_merge_transition_matrix_before.npy"
            T_after_path = result_dir / f"{stem}_merge_transition_matrix_after.npy"
            merge_map_path = result_dir / f"{stem}_merge_map.npy"
            np.save(T_before_path, info["T_before"])
            np.save(T_after_path, info["T_after"])
            np.save(merge_map_path, info["merge_map"])

            before_heatmap = result_dir / f"{stem}_merge_transition_matrix_before.png"
            after_heatmap = result_dir / f"{stem}_merge_transition_matrix_after.png"
            _save_transition_heatmap(info["T_before"], f"{stem} transitions before merge", before_heatmap)
            _save_transition_heatmap(info["T_after"], f"{stem} transitions after merge", after_heatmap)

            merge_summary = {
                "merge_applied": True,
                "k_before": int(info["k_before"]),
                "k_after": int(info["k_after"]),
                "merge_method": cfg.merge_method,
                "n_merge_target": int(cfg.n_merge),
                "used_boundary_mask": bool(boundary_mask is not None),
            }
            merge_info_paths = {
                "merged_plot": str(merged_plot),
                "merged_csv": str(merged_csv),
                "T_before_npy": str(T_before_path),
                "T_after_npy": str(T_after_path),
                "merge_map_npy": str(merge_map_path),
                "T_before_png": str(before_heatmap),
                "T_after_png": str(after_heatmap),
            }

    summary = {
        "npz_path": str(npz_path),
        "n_points_total": int(n_total),
        "n_points_used": int(X.shape[0]),
        "array_key": cfg.array_key,
        "plot_key": cfg.plot_key,
        "method_requested": cfg.method,
        "method_used": chosen_method,
        "n_clusters_requested": int(cfg.n_clusters),
        "n_neighbors": int(cfg.n_neighbors),
        "raw_plot": str(raw_plot),
        "raw_csv": str(raw_csv),
        **merge_summary,
    }

    summary_json = result_dir / f"{stem}_spectral_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    if cfg.save_augmented_npz:
        augmented_npz = result_dir / f"{stem}_spectral_outputs.npz"
        payload = {
            "original_row_index": original_indices,
            cfg.output_label_key: labels,
        }
        if merged_labels is not None:
            payload[cfg.merged_output_label_key] = merged_labels
        if file_indices is not None:
            payload[cfg.file_index_key] = file_indices
        np.savez_compressed(augmented_npz, **payload)
    else:
        augmented_npz = None

    result = {
        "result_dir": str(result_dir),
        "summary_json": str(summary_json),
        "raw_plot": str(raw_plot),
        "raw_csv": str(raw_csv),
    }
    if augmented_npz is not None:
        result["augmented_npz"] = str(augmented_npz)
    result.update(merge_info_paths)

    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run spectral clustering on one NPZ and optionally merge over-segmented sequential labels using transition-based merging."
    )
    parser.add_argument("--npz-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--array-key", type=str, default="predictions")
    parser.add_argument("--plot-key", type=str, default="embedding_outputs")
    parser.add_argument("--file-index-key", type=str, default="file_indices")
    parser.add_argument("--output-label-key", type=str, default="spectral_labels")
    parser.add_argument("--merged-output-label-key", type=str, default="spectral_labels_merged")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "gpu", "nystrom", "twostage"])
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-landmarks", type=int, default=5000)
    parser.add_argument("--n-subsample", type=int, default=10000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--palette-json", type=str, default=None)
    parser.add_argument("--drop-nonfinite-rows", action="store_true")
    parser.add_argument("--save-augmented-npz", action="store_true")

    parser.add_argument("--n-merge", type=int, default=None, help="Target number of merged clusters after spectral clustering.")
    parser.add_argument("--merge-method", type=str, default="average")
    parser.add_argument("--merge-seq-len", type=int, default=None)
    parser.add_argument("--no-merge-boundary-mask", action="store_true", help="Do not build a boundary mask from file_indices.")

    args = parser.parse_args()

    cfg = SpectralMergeConfig(
        array_key=args.array_key,
        plot_key=args.plot_key,
        file_index_key=args.file_index_key,
        output_label_key=args.output_label_key,
        merged_output_label_key=args.merged_output_label_key,
        method=args.method,
        n_clusters=args.n_clusters,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
        n_landmarks=args.n_landmarks,
        n_subsample=args.n_subsample,
        point_size=args.point_size,
        alpha=args.alpha,
        drop_nonfinite_rows=args.drop_nonfinite_rows,
        save_augmented_npz=args.save_augmented_npz,
        palette_json=args.palette_json,
        n_merge=args.n_merge,
        merge_method=args.merge_method,
        merge_use_boundary_mask=not args.no_merge_boundary_mask,
        merge_seq_len=args.merge_seq_len,
    )

    run_spectral_for_npz_with_merge(
        npz_path=args.npz_path,
        out_dir=args.out_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
