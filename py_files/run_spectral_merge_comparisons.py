#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_path = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_decoder/py_files/spectral_cluster_npz_with_transition_merge_compat.py")
    npz_path = Path("/Users/mirandahulsey-vincent/Desktop/USA5288.npz")
    palette_json = Path("/Users/mirandahulsey-vincent/Desktop/fixed_label_colors_50.json")
    base_out_dir = Path("/Users/mirandahulsey-vincent/Desktop/spectral_merge_comparisons")

    comparisons = [
        {"n_clusters": 40, "n_merge": 20, "name": "clusters40_merge20"},
        {"n_clusters": 40, "n_merge": 25, "name": "clusters40_merge25"},
        {"n_clusters": 50, "n_merge": 20, "name": "clusters50_merge20"},
    ]

    for cfg in comparisons:
        out_dir = base_out_dir / cfg["name"]
        cmd = [
            sys.executable,
            str(script_path),
            "--npz-path", str(npz_path),
            "--out-dir", str(out_dir),
            "--array-key", "predictions",
            "--plot-key", "embedding_outputs",
            "--file-index-key", "file_indices",
            "--n-clusters", str(cfg["n_clusters"]),
            "--method", "auto",
            "--n-neighbors", "30",
            "--n-merge", str(cfg["n_merge"]),
            "--merge-method", "average",
            "--palette-json", str(palette_json),
            "--save-augmented-npz",
        ]

        print("\n" + "=" * 80)
        print(f"Running comparison: {cfg['name']}")
        print(" ".join(cmd))
        print("=" * 80 + "\n")
        subprocess.run(cmd, check=True)

    print("\nAll comparisons completed.")
    print(f"Outputs saved under: {base_out_dir}")


if __name__ == "__main__":
    main()
