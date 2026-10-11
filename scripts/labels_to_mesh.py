"""Transfer per-point predictions back onto the mesh faces (majority vote).

Usage:
    python labels_to_mesh.py building.obj face_index.npy pred.npy building_segmented.ply

Faces that received no sample take the label of the nearest labelled face.
The output .ply has faces coloured by class and a per-face "label" property.
"""
import argparse

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [128, 128, 128],
    [0, 128, 128], [170, 110, 40], [128, 0, 0], [0, 0, 128], [255, 250, 200],
], np.uint8)


def colors_for(labels):
    c = PALETTE[np.asarray(labels) % len(PALETTE)].copy()
    c[np.asarray(labels) < 0] = 0
    return c


def face_labels(mesh, face_index, pred):
    """Majority vote of point labels per face; empty faces copy the nearest labelled face."""
    pred = np.asarray(pred).astype(np.int64)
    ok = pred >= 0
    n_faces, n_cls = len(mesh.faces), max(int(pred.max()) + 1, 1)
    votes = np.zeros((n_faces, n_cls), np.int64)
    np.add.at(votes, (face_index[ok], pred[ok]), 1)
    label = votes.argmax(1)
    empty = votes.sum(1) == 0
    if empty.any() and (~empty).any():
        centers = mesh.triangles_center
        _, nn = cKDTree(centers[~empty]).query(centers[empty])
        label[empty] = label[~empty][nn]
    return label, int(empty.sum())


def write_labelled_mesh(mesh, label, path):
    from plyfile import PlyData, PlyElement

    vertex = np.empty(len(mesh.vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = np.asarray(mesh.vertices, np.float32).T
    rgb = colors_for(label)
    face = np.empty(len(mesh.faces), dtype=[("vertex_indices", "i4", (3,)), ("label", "i4"),
                                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    face["vertex_indices"] = mesh.faces
    face["label"] = label
    face["red"], face["green"], face["blue"] = rgb.T
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")]).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("face_index")
    ap.add_argument("pred")
    ap.add_argument("output")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    face_index = np.load(args.face_index)
    pred = np.load(args.pred)
    if pred.ndim > 1:  # logits / probabilities
        pred = pred.argmax(1)
    assert len(pred) == len(face_index), "prediction count must match sampled points"
    label, filled = face_labels(mesh, face_index, pred)
    write_labelled_mesh(mesh, label, args.output)
    print(f"{len(label)} faces labelled ({filled} filled from neighbours) -> {args.output}")


if __name__ == "__main__":
    main()
