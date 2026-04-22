#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_sample_spectrograms_from_labels.py

Generate stitched sample spectrograms for arbitrary cluster labels aligned to an
NPZ spectrogram time axis. This is designed to work with labels produced by the
spectral-clustering and PCA->HDBSCAN pipelines, including labels saved in:

- augmented NPZ files with:
    - original_row_index
    - spectral_labels / spectral_labels_merged / similar
- CSV files with:
    - original_row_index
    - label

It can also make an embedding scatter colored by the chosen labels.

Main public function
--------------------
plot_spectrogram_samples_for_cluster_labels(
    source_npz_path,
    labels_source_path=None,
    output_dir=None,
    labels_key="spectral_labels_merged",
    ...
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict, List, Union, Tuple
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def _get_tab60_palette() -> List[str]:
    tab20 = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors
    return [mcolors.to_hex(c) for c in (*tab20, *tab20b, *tab20c)]


def build_label_color_lut(
    all_labels: np.ndarray,
    label_universe: Optional[Sequence[int]] = None,
    missing_label_color: str = "#bdbdbd",
) -> Dict[int, str]:
    palette = _get_tab60_palette()

    if label_universe is None:
        uniq = sorted(np.unique(all_labels.astype(int)))
    else:
        uniq = sorted({int(l) for l in label_universe} | {int(l) for l in np.unique(all_labels)})

    non_noise = [l for l in uniq if l != -1]
    lut: Dict[int, str] = {-1: "#7f7f7f"}

    for lab in non_noise:
        lut[lab] = palette[int(lab) % len(palette)]

    lut[-999] = missing_label_color
    return lut


def load_label_color_lut_from_json(
    json_path: Union[str, Path],
    all_labels: np.ndarray,
    label_universe: Optional[Sequence[int]] = None,
    missing_label_color: str = "#bdbdbd",
) -> Dict[int, str]:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object mapping labels to colors in {json_path}")

    fallback_lut = build_label_color_lut(
        all_labels=all_labels,
        label_universe=label_universe,
        missing_label_color=missing_label_color,
    )
    lut: Dict[int, str] = {}

    for key, value in raw.items():
        lut[int(key)] = str(value)

    for lab, color in fallback_lut.items():
        if lab not in lut:
            lut[lab] = color

    if -1 not in lut:
        lut[-1] = "#7f7f7f"
    if -999 not in lut:
        lut[-999] = missing_label_color
    return lut


def _infer_animal_id_from_path(p: Path) -> str:
    candidates: List[str] = []

    stem = p.stem.strip()
    if stem:
        parts = stem.split("_")
        candidates.append(parts[0].strip())
        candidates.append(stem)

    parent_name = p.parent.name.strip()
    if parent_name:
        candidates.append(parent_name)
        parent_parts = parent_name.split("_")
        if parent_parts:
            candidates.append(parent_parts[0].strip())

    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)

    return uniq[0] if uniq else p.stem


def _orient_spectrogram(S: np.ndarray, n_timebins: int) -> np.ndarray:
    if S.ndim != 2:
        raise ValueError("Expected a 2D spectrogram array in arr['s'].")

    if S.shape[1] == n_timebins:
        return S
    if S.shape[0] == n_timebins:
        return S.T

    raise ValueError(
        f"Could not align spectrogram shape {S.shape} with {n_timebins} timebins."
    )


def _load_labels_from_csv(
    csv_path: Path,
    n_timebins: int,
    label_col: str = "label",
    index_col: str = "original_row_index",
    missing_label_value: int = -999,
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"{label_col!r} not found in {csv_path}")
    if index_col not in df.columns:
        raise ValueError(f"{index_col!r} not found in {csv_path}")

    labels_full = np.full(n_timebins, missing_label_value, dtype=int)
    idx = df[index_col].to_numpy(dtype=int)
    vals = df[label_col].to_numpy(dtype=int)

    valid = (idx >= 0) & (idx < n_timebins)
    labels_full[idx[valid]] = vals[valid]
    return labels_full


def _load_labels_from_npz(
    npz_path: Path,
    n_timebins: int,
    labels_key: str,
    index_key: str = "original_row_index",
    missing_label_value: int = -999,
) -> np.ndarray:
    arr = np.load(npz_path, allow_pickle=True)
    try:
        if labels_key not in arr.files:
            raise ValueError(f"{labels_key!r} not found in {npz_path}. Keys: {list(arr.files)}")

        labels = np.asarray(arr[labels_key]).astype(int)

        if index_key in arr.files:
            idx = np.asarray(arr[index_key]).astype(int)
            labels_full = np.full(n_timebins, missing_label_value, dtype=int)
            valid = (idx >= 0) & (idx < n_timebins)
            labels_full[idx[valid]] = labels[valid]
            return labels_full

        if labels.shape[0] != n_timebins:
            raise ValueError(
                f"Labels length {labels.shape[0]} does not match spectrogram timebins {n_timebins}, "
                f"and {index_key!r} was not found for alignment."
            )
        return labels
    finally:
        try:
            arr.close()
        except Exception:
            pass


def _load_labels_any(
    labels_source_path: Optional[Union[str, Path]],
    source_npz_arr,
    n_timebins: int,
    labels_key: str,
    index_key: str = "original_row_index",
    label_col: str = "label",
    csv_index_col: str = "original_row_index",
    missing_label_value: int = -999,
) -> Tuple[np.ndarray, str]:
    if labels_source_path is None:
        if labels_key not in source_npz_arr.files:
            raise ValueError(
                f"labels_source_path not provided and {labels_key!r} not present in source NPZ. "
                f"Available keys: {list(source_npz_arr.files)}"
            )
        labels = np.asarray(source_npz_arr[labels_key]).astype(int)
        if labels.shape[0] != n_timebins:
            raise ValueError(
                f"Labels from source NPZ key {labels_key!r} have length {labels.shape[0]}, "
                f"but spectrogram has {n_timebins} timebins."
            )
        return labels, f"source NPZ key {labels_key}"

    labels_source_path = Path(labels_source_path)
    suffix = labels_source_path.suffix.lower()

    if suffix == ".csv":
        labels = _load_labels_from_csv(
            csv_path=labels_source_path,
            n_timebins=n_timebins,
            label_col=label_col,
            index_col=csv_index_col,
            missing_label_value=missing_label_value,
        )
        return labels, f"CSV {labels_source_path.name}"

    if suffix == ".npz":
        labels = _load_labels_from_npz(
            npz_path=labels_source_path,
            n_timebins=n_timebins,
            labels_key=labels_key,
            index_key=index_key,
            missing_label_value=missing_label_value,
        )
        return labels, f"NPZ {labels_source_path.name} key {labels_key}"

    raise ValueError("labels_source_path must be None, a .csv, or a .npz file.")


def plot_embedding_colored_by_labels(
    embedding: np.ndarray,
    labels: np.ndarray,
    lut: Dict[int, str],
    outdir: Optional[Path],
    show_plot: bool,
    add_legend: bool,
    *,
    animal_id: Optional[str] = None,
    title_suffix: str = "",
) -> None:
    if embedding.ndim != 2 or embedding.shape[0] != labels.shape[0] or embedding.shape[1] < 2:
        print("[WARN] Embedding plot skipped: embedding must be shape (T, 2+) matching labels length.")
        return

    colors = [lut[int(lab)] for lab in labels]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=10,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")

    if animal_id is not None:
        ax.set_title(f"{animal_id} — embedding colored by labels{title_suffix}")
    else:
        ax.set_title(f"Embedding colored by labels{title_suffix}")

    ax.grid(True, alpha=0.3)

    if add_legend:
        uniq = sorted(np.unique(labels.astype(int)))
        handles = [
            Patch(facecolor=lut[l], label=("missing" if l == -999 else str(l)))
            for l in uniq
            if l != -999
        ]

        max_rows = 14
        ncols = int(np.ceil(len(handles) / max_rows)) or 1

        ax.legend(
            handles=handles,
            title="Label",
            ncol=ncols,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=1.0,
            labelspacing=0.4,
        )
        fig.subplots_adjust(right=0.80)

    plt.tight_layout()

    if outdir is not None:
        save_path = outdir / "embedding_labels.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _pick_sample_indices(
    idx: np.ndarray,
    spectrogram_length: int,
    num_sample_spectrograms: int,
    random_seed: Optional[int] = None,
    randomize: bool = False,
) -> List[np.ndarray]:
    max_full_samples = idx.size // spectrogram_length
    actual_samples = min(num_sample_spectrograms, max_full_samples)
    if actual_samples == 0:
        return []

    chunks: List[np.ndarray] = []
    if not randomize:
        for k in range(actual_samples):
            start = k * spectrogram_length
            end = start + spectrogram_length
            chunks.append(idx[start:end])
        return chunks

    rng = np.random.default_rng(random_seed)
    possible_starts = np.arange(0, idx.size - spectrogram_length + 1)
    if possible_starts.size == 0:
        return []

    chosen_starts = rng.choice(
        possible_starts,
        size=actual_samples,
        replace=False if actual_samples <= possible_starts.size else True,
    )
    chosen_starts = np.sort(chosen_starts)
    for start in chosen_starts:
        chunks.append(idx[start:start + spectrogram_length])
    return chunks


def plot_spectrogram_samples_for_cluster_labels(
    source_npz_path: Union[str, Path],
    labels_source_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    labels_key: str = "spectral_labels_merged",
    label_col: str = "label",
    index_key: str = "original_row_index",
    csv_index_col: str = "original_row_index",
    selected_labels: Optional[Sequence[int]] = None,
    skip_noise_label: bool = True,
    skip_missing_label: bool = True,
    missing_label_value: int = -999,
    spectrogram_length: int = 1000,
    num_sample_spectrograms: int = 1,
    randomize_samples: bool = False,
    random_seed: Optional[int] = None,
    cmap: str = "gray_r",
    show_colorbar: bool = False,
    show_plots: bool = True,
    save_sample_spectrograms: bool = True,
    make_embedding_plot: bool = True,
    show_embedding_legend: bool = True,
    label_universe: Optional[Sequence[int]] = None,
    fixed_label_colors_json: Optional[Union[str, Path]] = None,
) -> None:
    source_npz_path = Path(source_npz_path)
    animal_id = _infer_animal_id_from_path(source_npz_path)
    arr = np.load(source_npz_path, allow_pickle=True)

    try:
        if "s" not in arr.files:
            raise ValueError(f"'s' not found in {source_npz_path}. Keys: {list(arr.files)}")

        S_raw = arr["s"]
        n_timebins_guess = None

        if "embedding_outputs" in arr.files:
            n_timebins_guess = arr["embedding_outputs"].shape[0]
        elif S_raw.ndim == 2:
            n_timebins_guess = max(S_raw.shape)
        else:
            raise ValueError("Could not infer spectrogram time length.")

        S = _orient_spectrogram(S_raw, n_timebins=n_timebins_guess)
        F, T = S.shape

        labels_full, labels_desc = _load_labels_any(
            labels_source_path=labels_source_path,
            source_npz_arr=arr,
            n_timebins=T,
            labels_key=labels_key,
            index_key=index_key,
            label_col=label_col,
            csv_index_col=csv_index_col,
            missing_label_value=missing_label_value,
        )

        embedding = arr["embedding_outputs"] if "embedding_outputs" in arr.files else None

        unique_labels = np.unique(labels_full)
        if selected_labels is None:
            labels_to_process = [int(l) for l in unique_labels]
            if skip_noise_label and (-1 in labels_to_process):
                labels_to_process.remove(-1)
            if skip_missing_label and (missing_label_value in labels_to_process):
                labels_to_process.remove(missing_label_value)
        else:
            labels_to_process = [int(l) for l in selected_labels]

        labels_to_process = sorted(labels_to_process)
        if not labels_to_process:
            print("[WARN] No labels selected to process.")
            return

        outdir: Optional[Path] = None
        if output_dir is not None:
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)

        if fixed_label_colors_json is not None:
            label_color_lut = load_label_color_lut_from_json(
                fixed_label_colors_json,
                labels_full,
                label_universe=label_universe,
                missing_label_color="#bdbdbd",
            )
            print(f"[INFO] Loaded fixed label colors from {fixed_label_colors_json}")
        else:
            label_color_lut = build_label_color_lut(
                labels_full,
                label_universe=label_universe,
                missing_label_color="#bdbdbd",
            )
            print("[INFO] Using internally generated label colors")

        print(f"[INFO] Using labels from {labels_desc}")
        print(f"[INFO] Spectrogram shape aligned to (freq, time) = {S.shape}")

        saved_paths: List[str] = []

        for lbl in labels_to_process:
            idx = np.flatnonzero(labels_full == lbl)
            if idx.size == 0:
                print(f"[WARN] No timebins for label {lbl}; skipping.")
                continue

            selected_chunks = _pick_sample_indices(
                idx=idx,
                spectrogram_length=spectrogram_length,
                num_sample_spectrograms=num_sample_spectrograms,
                random_seed=random_seed,
                randomize=randomize_samples,
            )

            if len(selected_chunks) == 0:
                print(
                    f"[WARN] Label {lbl}: only {idx.size} bins; need {spectrogram_length} "
                    f"for one sample. Skipping."
                )
                continue

            print(
                f"[INFO] {animal_id} — label {lbl}: making {len(selected_chunks)} sample(s) "
                f"of {spectrogram_length} bins each (bins available: {idx.size})."
            )

            for k, selected_idx in enumerate(selected_chunks, start=1):
                S_sel = S[:, selected_idx].astype(float)

                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.imshow(S_sel, origin="lower", aspect="auto", cmap=cmap)

                if show_colorbar:
                    fig.colorbar(im, ax=ax, label="Spectrogram")

                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False,
                )

                ax.set_title(
                    f"{animal_id} — label {lbl} — sample {k}/{len(selected_chunks)} "
                    f"({spectrogram_length} bins stitched)"
                )
                fig.tight_layout()

                if save_sample_spectrograms and outdir is not None:
                    fname = f"label{lbl}_sample{k}_N{spectrogram_length}.png"
                    save_path = outdir / fname
                    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
                    saved_paths.append(str(save_path))
                    print(f"[SAVE] {save_path}")

                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)

        if make_embedding_plot:
            if embedding is None:
                print("[WARN] 'embedding_outputs' not found in source NPZ; skipping embedding plot.")
            else:
                plot_embedding_colored_by_labels(
                    embedding=embedding,
                    labels=labels_full,
                    lut=label_color_lut,
                    outdir=outdir,
                    show_plot=show_plots,
                    add_legend=show_embedding_legend,
                    animal_id=animal_id,
                    title_suffix=f" ({labels_key})" if labels_source_path is not None else "",
                )

        if saved_paths:
            print("\nAll saved sample spectrogram files:")
            for p in saved_paths:
                print("  ", p)

    finally:
        try:
            arr.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stitched sample spectrograms for cluster labels aligned to an NPZ spectrogram."
    )
    parser.add_argument("--source-npz", required=True, type=str, help="Original NPZ containing spectrogram key 's'.")
    parser.add_argument("--labels-source", default=None, type=str, help="Optional labels source (.npz or .csv).")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--labels-key", default="spectral_labels_merged", type=str)
    parser.add_argument("--label-col", default="label", type=str)
    parser.add_argument("--index-key", default="original_row_index", type=str)
    parser.add_argument("--csv-index-col", default="original_row_index", type=str)
    parser.add_argument("--selected-labels", default=None, type=str, help="Comma-separated labels to plot.")
    parser.add_argument("--skip-noise-label", action="store_true")
    parser.add_argument("--skip-missing-label", action="store_true")
    parser.add_argument("--spectrogram-length", default=1000, type=int)
    parser.add_argument("--num-sample-spectrograms", default=1, type=int)
    parser.add_argument("--randomize-samples", action="store_true")
    parser.add_argument("--random-seed", default=None, type=int)
    parser.add_argument("--cmap", default="gray_r", type=str)
    parser.add_argument("--show-colorbar", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--make-embedding-plot", action="store_true")
    parser.add_argument("--show-embedding-legend", action="store_true")
    parser.add_argument("--fixed-label-colors-json", default=None, type=str)
    args = parser.parse_args()

    selected_labels = None
    if args.selected_labels:
        selected_labels = [int(x.strip()) for x in args.selected_labels.split(",") if x.strip()]

    plot_spectrogram_samples_for_cluster_labels(
        source_npz_path=args.source_npz,
        labels_source_path=args.labels_source,
        output_dir=args.output_dir,
        labels_key=args.labels_key,
        label_col=args.label_col,
        index_key=args.index_key,
        csv_index_col=args.csv_index_col,
        selected_labels=selected_labels,
        skip_noise_label=args.skip_noise_label,
        skip_missing_label=args.skip_missing_label,
        spectrogram_length=args.spectrogram_length,
        num_sample_spectrograms=args.num_sample_spectrograms,
        randomize_samples=args.randomize_samples,
        random_seed=args.random_seed,
        cmap=args.cmap,
        show_colorbar=args.show_colorbar,
        show_plots=args.show_plots,
        save_sample_spectrograms=True,
        make_embedding_plot=args.make_embedding_plot,
        show_embedding_legend=args.show_embedding_legend,
        fixed_label_colors_json=args.fixed_label_colors_json,
    )


if __name__ == "__main__":
    main()
