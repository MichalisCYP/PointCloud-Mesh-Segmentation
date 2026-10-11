"""Transfer per-point predictions back onto the mesh faces (majority vote).

Usage:
    python labels_to_mesh.py building.obj data/heritage/test/building01/face_index.npy \
        exp/heritage/result/building01_pred.npy building01_segmented.ply

Faces that received no sample take the label of the nearest labelled face.
The output .ply has faces coloured by class and a per-face "label" property.
"""
import argparse

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("face_index")
    ap.add_argument("pred")
    ap.add_argument("output")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    face_index = np.load(args.face_index)
    pred = np.load(args.pred).astype(np.int64)
    if pred.ndim > 1:  # logits / probabilities
        pred = pred.argmax(1)
    assert len(pred) == len(face_index), "prediction count must match sampled points"

    n_faces, n_cls = len(mesh.faces), pred.max() + 1
    votes = np.zeros((n_faces, n_cls), np.int64)
    np.add.at(votes, (face_index, pred), 1)
    face_label = votes.argmax(1)

    empty = votes.sum(1) == 0
    if empty.any() and (~empty).any():
        centers = mesh.triangles_center
        _, nn = cKDTree(centers[~empty]).query(centers[empty])
        face_label[empty] = face_label[~empty][nn]

    palette = (np.random.default_rng(0).random((max(n_cls, 1), 3)) * 255).astype(np.uint8)

    from plyfile import PlyData, PlyElement

    vertex = np.array([tuple(v) for v in mesh.vertices], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    rgb = palette[face_label]
    face = np.empty(n_faces, dtype=[("vertex_indices", "i4", (3,)), ("label", "i4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    face["vertex_indices"] = mesh.faces
    face["label"] = face_label
    face["red"], face["green"], face["blue"] = rgb.T
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")]).write(args.output)
    print(f"{n_faces} faces labelled ({empty.sum()} filled from neighbours) -> {args.output}")


if __name__ == "__main__":
    main()
