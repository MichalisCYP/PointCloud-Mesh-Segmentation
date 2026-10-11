"""Convert a .ply / .las / .laz point cloud into Pointcept's per-scene .npy format.

Usage:
    python prepare_pointcloud.py scan.las data/heritage/train/scan01 [--label-field classification]

Writes coord.npy (float32), color.npy (uint8 0-255, if present) and
segment.npy (int32, -1 = unlabelled) into the output directory.
"""
import argparse
from pathlib import Path

import numpy as np


def read_ply(path, label_field):
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    names = v.dtype.names
    coord = np.stack([v["x"], v["y"], v["z"]], 1)
    color = None
    if {"red", "green", "blue"} <= set(names):
        color = np.stack([v["red"], v["green"], v["blue"]], 1)
    label = np.asarray(v[label_field]) if label_field and label_field in names else None
    return coord, color, label


def read_las(path, label_field):
    import laspy

    las = laspy.read(str(path))
    coord = np.stack([las.x, las.y, las.z], 1)
    color = None
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        rgb = np.stack([las.red, las.green, las.blue], 1)
        color = (rgb >> 8) if rgb.max() > 255 else rgb  # LAS stores 16-bit colour
    label = np.asarray(las[label_field]) if label_field else None
    return coord, color, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--label-field", default=None,
                    help="per-point label attribute (e.g. classification, scalar_label)")
    args = ap.parse_args()

    src = Path(args.input)
    reader = read_las if src.suffix.lower() in {".las", ".laz"} else read_ply
    coord, color, label = reader(src, args.label_field)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Large georeferenced coordinates lose precision in float32, so centre first.
    coord = coord - coord.mean(0)
    np.save(out / "coord.npy", coord.astype(np.float32))
    if color is not None:
        np.save(out / "color.npy", np.asarray(color).astype(np.uint8))
    if label is not None:
        np.save(out / "segment.npy", label.astype(np.int32))
    print(f"{src.name}: {len(coord)} points, color={color is not None}, labels={label is not None} -> {out}")


if __name__ == "__main__":
    main()
