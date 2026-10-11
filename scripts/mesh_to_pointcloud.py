"""Sample a (textured) mesh into Pointcept format, remembering which face each
point came from so predictions can be voted back onto the mesh.

Example:
    python mesh_to_pointcloud.py facade.obj data/heritage/test --points 3000000 --tile 10

Writes the scene like prepare_pointcloud.py and saves face_index.npy under
data/heritage/meta/<name>/. A .ply mesh with a per-face integer "label"
property also yields segment.npy, so labelled meshes can be used for training.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh

from common import save_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("split_dir")
    ap.add_argument("--points", type=int, default=3_000_000)
    ap.add_argument("--tile", type=float, default=None)
    ap.add_argument("--buffer", type=float, default=2.0)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    try:
        coord, face_index, color = trimesh.sample.sample_surface(mesh, args.points, sample_color=True)
        color = np.asarray(color)[:, :3]
    except Exception:  # no usable texture or vertex colours
        coord, face_index = trimesh.sample.sample_surface(mesh, args.points)
        color = None

    segment = None
    if args.mesh.lower().endswith(".ply"):
        from plyfile import PlyData

        face = PlyData.read(args.mesh)["face"].data
        if "label" in face.dtype.names:
            segment = np.asarray(face["label"])[face_index]

    name = Path(args.mesh).stem
    n = save_scene(args.split_dir, name, coord, color, mesh.face_normals[face_index], segment,
                   args.tile, args.buffer, meta={"face_index": face_index.astype(np.int64)})
    print(f"{args.mesh}: {len(mesh.faces)} faces -> {len(coord)} points -> {n} scene folder(s)")


if __name__ == "__main__":
    main()
