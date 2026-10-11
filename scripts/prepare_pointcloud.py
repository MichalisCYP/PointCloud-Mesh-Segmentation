"""Convert a point cloud (.ply / .las / .laz / .txt) into Pointcept format.

Examples:
    # labelled LAS scan, tiled for an 8 GB GPU
    python prepare_pointcloud.py church.las data/heritage/train --label-field classification --tile 10

    # ArCH-style text file: x y z r g b label (column order given explicitly)
    python prepare_pointcloud.py scene.txt data/heritage/train --txt-cols x,y,z,r,g,b,label --tile 10

    # unlabelled scan to predict on
    python prepare_pointcloud.py cloister.ply data/heritage/test --tile 10

The scene name is the input file stem. Big TLS scans: add --voxel 0.01 to thin
them first (the model works at 2 cm anyway).
"""
import argparse
from pathlib import Path

import numpy as np

from common import save_scene, voxel_downsample


def read_ply(path, label_field, _):
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    names = set(v.dtype.names)
    coord = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    color = np.stack([v["red"], v["green"], v["blue"]], 1) if {"red", "green", "blue"} <= names else None
    normal = np.stack([v["nx"], v["ny"], v["nz"]], 1) if {"nx", "ny", "nz"} <= names else None
    label = np.asarray(v[label_field]) if label_field and label_field in names else None
    return coord, color, normal, label


def read_las(path, label_field, _):
    import laspy

    las = laspy.read(str(path))
    coord = np.stack([las.x, las.y, las.z], 1)
    dims = set(las.point_format.dimension_names)
    color = None
    if {"red", "green", "blue"} <= dims:
        rgb = np.stack([las.red, las.green, las.blue], 1)
        color = (rgb >> 8) if rgb.max() > 255 else rgb  # LAS colour is often 16-bit
    label = np.asarray(las[label_field]) if label_field else None
    return coord, color, None, label


def read_txt(path, _, cols):
    cols = cols.split(",")
    data = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    get = lambda *names: np.stack([data[:, cols.index(n)] for n in names], 1) if set(names) <= set(cols) else None
    coord = get("x", "y", "z")
    color = get("r", "g", "b")
    normal = get("nx", "ny", "nz")
    label = data[:, cols.index("label")] if "label" in cols else None
    return coord, color, normal, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("split_dir", help="e.g. data/heritage/train")
    ap.add_argument("--label-field", default=None, help="per-point label attribute in PLY/LAS")
    ap.add_argument("--txt-cols", default="x,y,z,r,g,b,label", help="column names for .txt/.csv")
    ap.add_argument("--voxel", type=float, default=None, help="pre-thin to this voxel size (m)")
    ap.add_argument("--tile", type=float, default=None, help="XY tile size in m (use ~10 on 8 GB)")
    ap.add_argument("--buffer", type=float, default=2.0, help="context border around each tile (m)")
    args = ap.parse_args()

    src = Path(args.input)
    ext = src.suffix.lower()
    reader = {".las": read_las, ".laz": read_las, ".txt": read_txt, ".csv": read_txt}.get(ext, read_ply)
    coord, color, normal, label = reader(src, args.label_field, args.txt_cols)
    print(f"{src.name}: {len(coord)} points")

    if args.voxel:
        keep = voxel_downsample(coord, args.voxel)
        coord, color, normal, label = (a[keep] if a is not None else None for a in (coord, color, normal, label))
        print(f"  thinned to {len(coord)} points at {args.voxel} m")

    n = save_scene(args.split_dir, src.stem, coord, color, normal, label, args.tile, args.buffer)
    print(f"  wrote {n} scene folder(s) to {args.split_dir} (labels={label is not None})")


if __name__ == "__main__":
    main()
