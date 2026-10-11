"""Sample a (textured) mesh into a point cloud in Pointcept format, remembering
which face each point came from so predictions can be mapped back to the mesh.

Usage:
    python mesh_to_pointcloud.py building.obj data/heritage/test/building01 [--points 2000000]

Writes coord.npy, color.npy (if the mesh has texture/vertex colours) and
face_index.npy. If the mesh is a .ply with a per-face integer "label"
property, segment.npy is written too (for training on labelled meshes).
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh


def face_labels_from_ply(path):
    from plyfile import PlyData

    face = PlyData.read(str(path))["face"].data
    return np.asarray(face["label"]) if "label" in face.dtype.names else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--points", type=int, default=2_000_000)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    try:
        coord, face_index, color = trimesh.sample.sample_surface(mesh, args.points, sample_color=True)
    except Exception:  # no usable texture or vertex colours
        coord, face_index = trimesh.sample.sample_surface(mesh, args.points)
        color = None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "coord.npy", (coord - mesh.vertices.mean(0)).astype(np.float32))
    np.save(out / "normal.npy", mesh.face_normals[face_index].astype(np.float32))
    np.save(out / "face_index.npy", face_index.astype(np.int64))
    if color is not None:
        np.save(out / "color.npy", np.asarray(color)[:, :3].astype(np.uint8))
    if args.mesh.lower().endswith(".ply"):
        labels = face_labels_from_ply(args.mesh)
        if labels is not None:
            np.save(out / "segment.npy", labels[face_index].astype(np.int32))
    print(f"{args.mesh}: {len(mesh.faces)} faces -> {len(coord)} points, color={color is not None} -> {out}")


if __name__ == "__main__":
    main()
