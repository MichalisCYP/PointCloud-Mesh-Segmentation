"""Stitch Pointcept per-tile predictions back into one prediction per scene.

Example:
    python merge_tiles.py data/heritage/test exp/heritage/sonata-ft/result

For every scene in the split dir, writes <result_dir>/merged/<scene>_pred.npy
with one label per original (prepared) point. Each point takes the prediction
from the tile whose core contains it, so the buffer only adds context.
Untiled scenes are copied through unchanged.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("result_dir")
    args = ap.parse_args()

    split, result = Path(args.split_dir), Path(args.result_dir)
    meta = split.parent / "meta"
    out = result / "merged"
    out.mkdir(exist_ok=True)

    scenes = defaultdict(list)
    for d in sorted(p for p in split.iterdir() if p.is_dir()):
        scenes[d.name.split("__t")[0]].append(d)

    for name, parts in scenes.items():
        n = int(np.load(meta / name / "num_points.npy"))
        pred = np.full(n, -1, np.int64)
        for d in parts:
            p = np.load(result / f"{d.name}_pred.npy")
            if (d / "orig_index.npy").exists():
                idx, core = np.load(d / "orig_index.npy"), np.load(d / "core.npy")
                pred[idx[core]] = p[core]
            else:
                pred[:] = p
        missing = (pred < 0).sum()
        np.save(out / f"{name}_pred.npy", pred)
        print(f"{name}: {len(parts)} part(s), {n} points" + (f", {missing} UNPREDICTED" if missing else ""))


if __name__ == "__main__":
    main()
