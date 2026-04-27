#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_cluster_npz_latent_space_discrete.py

Purpose
-------
Run spectral clustering on latent-space arrays stored inside .npz files,
using the GPU/torch-based API from:
https://github.com/timothyjgardner/gpu_spectral_clustering

What this script does
---------------------
For each NPZ:
  - Load a latent-space array from `array_key` (default: "predictions")
  - Run one of the gpu_spectral clustering methods:
      * GPUSpectral
      * NystromSpectral
      * TwoStageSpectral
      * or method="auto"
  - Optionally merge over-segmented clusters using transition structure
  - Save:
      * per-point label CSV
      * labels-only NPZ
      * cluster-count CSV
      * optional 2D scatter plot if a 2D plotting array is available
  - Optionally save an augmented NPZ containing the new labels

New in this version
-------------------
- Supports discrete cluster colors from a JSON file that maps label strings
  to hex colors, e.g. {"0": "#1f77b4", "1": "#aec7e8", ...}
- Uses a discrete legend instead of a continuous colorbar
- Supports either a single NPZ (--npz-path) or a directory of NPZs (--root-dir)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from gpu_spectral import GPUSpectral, NystromSpectral, TwoStageSpectral, merge_clusters
except ImportError as e:
    raise ImportError(
        "Could not import gpu_spectral. Install the repo first, for example with\n"
        "git clone https://github.com/timothyjgardner/gpu_spectral_clustering.git\n"
        "cd gpu_spectral_clustering\n"
        "pip install -e .\n"
    ) from e


# ============================================================
# Helpers
# ============================================================
def _safe_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _load_array(data: Any, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key {key!r}. Available keys: {list(data.keys())}")
    return _safe_2d(np.asarray(data[key]))


def _choose_method(method: str, n_points: int) -> str:
    method = str(method).lower()
    if method != "auto":
        return method
    if n_points < 100_000:
        return "gpu"
    if n_points < 2_000_000:
        return "nystrom"
    return "two_stage"


def _make_clusterer(cfg: "SpectralConfig", n_points: int):
    method_used = _choose_method(cfg.method, n_points)

    if method_used == "gpu":
        clusterer = GPUSpectral(
            n_clusters=cfg.n_clusters,
            n_neighbors=cfg.n_neighbors,
            seed=cfg.seed,
        )
    elif method_used == "nystrom":
        clusterer = NystromSpectral(
            n_clusters=cfg.n_clusters,
            n_neighbors=cfg.n_neighbors,
            seed=cfg.seed,
            n_landmarks=min(cfg.n_landmarks, n_points),
        )
    elif method_used in {"two_stage", "twostage"}:
        method_used = "two_stage"
        clusterer = TwoStageSpectral(
            n_clusters=cfg.n_clusters,
            n_neighbors=cfg.n_neighbors,
            seed=cfg.seed,
            n_subsample=min(cfg.n_subsample, n_points),
        )
    else:
        raise ValueError("method must be one of: auto, gpu, nystrom, two_stage")

    return clusterer, method_used


def _write_rows_csv(csv_path: str | Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("")
        return
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _sample_for_plot(X: np.ndarray, labels: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n <= max_points:
        return X, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return X[idx], labels[idx]


def _load_palette_json(palette_json: Optional[str | Path]) -> Dict[str, str]:
    if palette_json is None:
        return {}
    palette_path = Path(palette_json)
    with open(palette_path, "r") as f:
        raw = json.load(f)
    palette: Dict[str, str] = {}
    for k, v in raw.items():
        palette[str(k)] = str(v)
    return palette


def _fallback_color(index: int) -> str:
    cmap = plt.get_cmap("tab20")
    rgba = cmap(index % cmap.N)
    return matplotlib.colors.to_hex(rgba)


def _label_to_color(label: int, palette: Dict[str, str], order_index: int) -> str:
    if str(label) in palette:
        return palette[str(label)]
    return _fallback_color(order_index)


def plot_labels_2d(
    X_plot: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | Path,
    max_points: int = 200_000,
    seed: int = 42,
    palette_json: Optional[str | Path] = None,
    legend_marker_size: float = 6.0,
) -> None:
    X_plot = _safe_2d(X_plot)
    if X_plot.shape[1] != 2:
        return

    Xp, lp = _sample_for_plot(X_plot, labels, max_points=max_points, seed=seed)
    palette = _load_palette_json(palette_json)

    unique_labels = np.unique(lp)
    unique_labels = np.sort(unique_labels)
    color_map: Dict[int, str] = {
        int(label): _label_to_color(int(label), palette, idx)
        for idx, label in enumerate(unique_labels)
    }
    point_colors = [color_map[int(label)] for label in lp]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Xp[:, 0], Xp[:, 1], c=point_colors, s=3, alpha=0.7, linewidths=0)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=legend_marker_size,
            markerfacecolor=color_map[int(label)],
            markeredgecolor=color_map[int(label)],
            label=str(int(label)),
        )
        for label in unique_labels
    ]

    if handles:
        ncol = 1 if len(handles) <= 20 else 2
        ax.legend(
            handles=handles,
            title="Spectral cluster",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            ncol=ncol,
            borderaxespad=0.0,
        )

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Spectral clustering core
# ============================================================
@dataclass
class SpectralConfig:
    array_key: str = "predictions"
    plot_key: Optional[str] = "embedding_outputs"
    output_label_key: str = "spectral_labels"

    method: str = "auto"
    n_clusters: int = 20
    n_neighbors: int = 30
    seed: int = 42
    n_landmarks: int = 5000
    n_subsample: int = 10000

    merge_to_n_clusters: Optional[int] = None
    seq_len: Optional[int] = None

    drop_nonfinite_rows: bool = True
    save_augmented_npz: bool = False
    plot_max_points: int = 200_000
    palette_json: Optional[str] = None
    show: bool = False


def run_spectral_for_npz(
    npz_path: str | Path,
    out_dir: str | Path,
    cfg: SpectralConfig,
) -> Dict[str, str]:
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = npz_path.stem

    data = np.load(npz_path, allow_pickle=True)
    X_all = _load_array(data, cfg.array_key)
    n_total, n_dim = X_all.shape

    if cfg.drop_nonfinite_rows:
        valid_mask = np.all(np.isfinite(X_all), axis=1)
    else:
        valid_mask = np.ones(n_total, dtype=bool)

    if not np.any(valid_mask):
        raise ValueError(f"No valid rows available in {npz_path}")

    X_valid = np.asarray(X_all[valid_mask], dtype=np.float32)
    n_valid = X_valid.shape[0]

    if n_valid < cfg.n_clusters:
        raise ValueError(
            f"n_valid={n_valid} is smaller than n_clusters={cfg.n_clusters}. "
            "Reduce --n-clusters or choose a larger dataset."
        )

    clusterer, method_used = _make_clusterer(cfg, n_valid)
    labels_valid = np.asarray(clusterer.fit_predict(X_valid), dtype=int)
    labels_before_merge = labels_valid.copy()

    merge_info: Dict[str, Any] = {}
    if cfg.merge_to_n_clusters is not None:
        unique_before = np.unique(labels_valid)
        if 1 < cfg.merge_to_n_clusters < unique_before.size:
            merged_labels, info = merge_clusters(
                labels_valid,
                n_merge=int(cfg.merge_to_n_clusters),
                seq_len=cfg.seq_len,
            )
            labels_valid = np.asarray(merged_labels, dtype=int)
            merge_info = {
                "k_before": int(info.get("k_before", unique_before.size)),
                "k_after": int(info.get("k_after", np.unique(labels_valid).size)),
            }

    labels_all = np.full(n_total, -1, dtype=int)
    labels_all[valid_mask] = labels_valid

    labels_before_merge_all = np.full(n_total, -1, dtype=int)
    labels_before_merge_all[valid_mask] = labels_before_merge

    npz_out_dir = out_dir / base
    npz_out_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = npz_out_dir / f"{base}_{cfg.output_label_key}_{ts}.csv"
    labels_df = pd.DataFrame({
        "row_index": np.arange(n_total, dtype=int),
        "valid_for_clustering": valid_mask.astype(bool),
        f"{cfg.output_label_key}_before_merge": labels_before_merge_all,
        cfg.output_label_key: labels_all,
    })
    labels_df.to_csv(labels_csv, index=False)

    counts_csv = npz_out_dir / f"{base}_{cfg.output_label_key}_counts_{ts}.csv"
    counts = (
        pd.Series(labels_valid, name=cfg.output_label_key)
        .value_counts()
        .sort_index()
        .rename_axis("cluster_id")
        .reset_index(name="n_rows")
    )
    counts.to_csv(counts_csv, index=False)

    labels_npz = npz_out_dir / f"{base}_{cfg.output_label_key}_{ts}.npz"
    np.savez_compressed(
        labels_npz,
        row_index=np.arange(n_total, dtype=int),
        valid_mask=valid_mask,
        labels_before_merge=labels_before_merge_all,
        labels=labels_all,
        npz_path=str(npz_path),
        array_key=cfg.array_key,
        method_used=method_used,
    )

    plot_path = ""
    plot_source = ""
    X_plot = None
    if cfg.plot_key is not None and cfg.plot_key in data:
        candidate = _safe_2d(np.asarray(data[cfg.plot_key]))
        if candidate.shape[0] == n_total and candidate.shape[1] == 2:
            X_plot = candidate[valid_mask]
            plot_source = cfg.plot_key
    elif X_valid.shape[1] == 2:
        X_plot = X_valid
        plot_source = cfg.array_key

    if X_plot is not None:
        plot_path = str(npz_out_dir / f"{base}_{cfg.output_label_key}_plot_{ts}.png")
        plot_labels_2d(
            X_plot=X_plot,
            labels=labels_valid,
            title=f"{base} spectral clustering ({method_used})",
            save_path=plot_path,
            max_points=cfg.plot_max_points,
            seed=cfg.seed,
            palette_json=cfg.palette_json,
        )

    augmented_npz_path = ""
    if cfg.save_augmented_npz:
        augmented_npz_path = str(npz_out_dir / f"{base}_with_{cfg.output_label_key}_{ts}.npz")
        payload = {k: data[k] for k in data.files}
        payload[f"{cfg.output_label_key}_before_merge"] = labels_before_merge_all
        payload[cfg.output_label_key] = labels_all
        np.savez_compressed(augmented_npz_path, **payload)

    summary = {
        "npz_path": str(npz_path),
        "array_key": cfg.array_key,
        "plot_key": plot_source,
        "n_total_rows": int(n_total),
        "n_valid_rows": int(n_valid),
        "n_dimensions": int(n_dim),
        "method_requested": cfg.method,
        "method_used": method_used,
        "n_clusters_requested": int(cfg.n_clusters),
        "n_clusters_found": int(np.unique(labels_valid).size),
        "n_neighbors": int(cfg.n_neighbors),
        "seed": int(cfg.seed),
        "n_landmarks": int(min(cfg.n_landmarks, n_valid)),
        "n_subsample": int(min(cfg.n_subsample, n_valid)),
        "merge_to_n_clusters": ("" if cfg.merge_to_n_clusters is None else int(cfg.merge_to_n_clusters)),
        "seq_len": ("" if cfg.seq_len is None else int(cfg.seq_len)),
        "palette_json": ("" if cfg.palette_json is None else str(cfg.palette_json)),
        "labels_csv": str(labels_csv),
        "counts_csv": str(counts_csv),
        "labels_npz": str(labels_npz),
        "plot_path": plot_path,
        "augmented_npz_path": augmented_npz_path,
    }
    summary.update(merge_info)

    summary_json = npz_out_dir / f"{base}_{cfg.output_label_key}_summary_{ts}.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    summary["summary_json"] = str(summary_json)
    return summary


# ============================================================
# Root-directory batch runner
# ============================================================
def run_root_directory_spectral(
    root_dir: str | Path,
    out_dir: str | Path | None = None,
    recursive: bool = True,
    cfg: Optional[SpectralConfig] = None,
) -> str:
    root_dir = Path(root_dir)
    if out_dir is None:
        out_dir = root_dir / "spectral_cluster_outputs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        cfg = SpectralConfig()

    npz_paths = sorted(root_dir.rglob("*.npz") if recursive else root_dir.glob("*.npz"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    master_rows: List[Dict[str, Any]] = []

    for npz_path in npz_paths:
        print(f"[spectral] processing: {npz_path}")
        try:
            result = run_spectral_for_npz(npz_path=npz_path, out_dir=out_dir, cfg=cfg)
            result["error"] = ""
            master_rows.append(result)
            print(f"[spectral] done: {npz_path.name}")
        except Exception as e:
            master_rows.append({
                "npz_path": str(npz_path),
                "array_key": cfg.array_key,
                "method_requested": cfg.method,
                "n_clusters_requested": cfg.n_clusters,
                "palette_json": ("" if cfg.palette_json is None else str(cfg.palette_json)),
                "labels_csv": "",
                "counts_csv": "",
                "labels_npz": "",
                "plot_path": "",
                "augmented_npz_path": "",
                "summary_json": "",
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"[spectral] failed: {npz_path.name} ({type(e).__name__}: {e})")

    master_csv = out_dir / f"spectral_master_summary_{ts}.csv"
    _write_rows_csv(master_csv, master_rows)
    print(f"[spectral] MASTER summary saved: {master_csv}")
    return str(master_csv)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Run gpu_spectral clustering on NPZ latent-space arrays.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--root-dir", type=str, default=None)
    group.add_argument("--npz-path", type=str, default=None)

    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--recursive", action="store_true")

    p.add_argument("--array-key", type=str, default="predictions")
    p.add_argument("--plot-key", type=str, default="embedding_outputs")
    p.add_argument("--output-label-key", type=str, default="spectral_labels")
    p.add_argument("--palette-json", type=str, default=None)

    p.add_argument("--method", type=str, default="auto", choices=["auto", "gpu", "nystrom", "two_stage"])
    p.add_argument("--n-clusters", type=int, required=True)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-landmarks", type=int, default=5000)
    p.add_argument("--n-subsample", type=int, default=10000)

    p.add_argument("--merge-to-n-clusters", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)

    p.add_argument("--keep-nonfinite-rows", action="store_true")
    p.add_argument("--save-augmented-npz", action="store_true")
    p.add_argument("--plot-max-points", type=int, default=200000)

    args = p.parse_args()

    cfg = SpectralConfig(
        array_key=args.array_key,
        plot_key=(None if str(args.plot_key).lower() == "none" else args.plot_key),
        output_label_key=args.output_label_key,
        method=args.method,
        n_clusters=args.n_clusters,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
        n_landmarks=args.n_landmarks,
        n_subsample=args.n_subsample,
        merge_to_n_clusters=args.merge_to_n_clusters,
        seq_len=args.seq_len,
        drop_nonfinite_rows=(not args.keep_nonfinite_rows),
        save_augmented_npz=args.save_augmented_npz,
        plot_max_points=args.plot_max_points,
        palette_json=args.palette_json,
    )

    if args.npz_path is not None:
        npz_path = Path(args.npz_path)
        out_dir = Path(args.out_dir) if args.out_dir is not None else npz_path.parent / "spectral_cluster_outputs"
        result = run_spectral_for_npz(
            npz_path=npz_path,
            out_dir=out_dir,
            cfg=cfg,
        )
        print(json.dumps(result, indent=2))
        return

    master_csv = run_root_directory_spectral(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        recursive=args.recursive,
        cfg=cfg,
    )
    print(master_csv)


if __name__ == "__main__":
    main()


# =============================================================================
# Spyder Console Sample Usage (copy/paste)
# =============================================================================
# from pathlib import Path
# import sys, importlib
#
# code_dir = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class_decoder/py_files")
# if str(code_dir) not in sys.path:
#     sys.path.insert(0, str(code_dir))
#
# import spectral_cluster_npz_latent_space_discrete as scn
# importlib.reload(scn)
#
# npz_path = Path("/Volumes/my_own_SSD/updated_AreaX_outputs/USA5288/USA5288.npz")
#
# cfg = scn.SpectralConfig(
#     array_key="predictions",
#     plot_key="embedding_outputs",
#     output_label_key="spectral_labels",
#     palette_json="/Volumes/my_own_SSD/updated_AreaX_outputs/fixed_label_colors_50.json",
#     method="auto",
#     n_clusters=20,
#     n_neighbors=30,
#     seed=42,
#     n_landmarks=5000,
#     n_subsample=10000,
#     merge_to_n_clusters=None,
#     seq_len=None,
#     drop_nonfinite_rows=True,
#     save_augmented_npz=False,
#     plot_max_points=200000,
# )
#
# result = scn.run_spectral_for_npz(
#     npz_path=npz_path,
#     out_dir=npz_path.parent / "spectral_cluster_outputs",
#     cfg=cfg,
# )
# print(result)
