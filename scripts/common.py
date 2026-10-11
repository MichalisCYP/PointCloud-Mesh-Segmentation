"""Shared helpers for writing scenes in Pointcept's per-folder .npy format."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(coord, size):
    """Indices of one point per voxel of `size` metres (keeps the first hit)."""
    keys = np.floor((coord - coord.min(0)) / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


def estimate_normals(coord, k=16, chunk=500_000):
    """PCA normals from k nearest neighbours, oriented to point +Z-ish/outwards-ish."""
    tree = cKDTree(coord)
    normals = np.empty_like(coord, dtype=np.float32)
    for s in range(0, len(coord), chunk):
        _, nn = tree.query(coord[s:s + chunk], k=k, workers=-1)
        nb = coord[nn] - coord[nn].mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        _, vec = np.linalg.eigh(cov)
        normals[s:s + chunk] = vec[:, :, 0]  # eigenvector of smallest eigenvalue
    # Consistent orientation: flip towards the scene centroid's opposite side.
    outward = coord - coord.mean(0)
    normals[(normals * outward).sum(1) < 0] *= -1
    return normals


def tiles(coord, size, buffer):
    """Yield (indices, core_mask) for XY tiles of `size` m with a `buffer` m context border."""
    lo = coord[:, :2].min(0)
    cell = np.floor((coord[:, :2] - lo) / size).astype(np.int64)
    for cx, cy in np.unique(cell, axis=0):
        x0, y0 = lo + np.array([cx, cy]) * size
        inside = ((coord[:, 0] >= x0 - buffer) & (coord[:, 0] < x0 + size + buffer) &
                  (coord[:, 1] >= y0 - buffer) & (coord[:, 1] < y0 + size + buffer))
        idx = np.flatnonzero(inside)
        core = (cell[idx, 0] == cx) & (cell[idx, 1] == cy)
        yield idx, core


def save_scene(out_dir, name, coord, color=None, normal=None, segment=None,
               tile_size=None, buffer=2.0, meta=None):
    """Write <out_dir>/<name>[__tNNN]/{coord,color,normal,segment}.npy.

    Pointcept needs colour and normals for Sonata (9 input channels), so missing
    colour is filled with zeros and missing normals are estimated. With
    `tile_size`, large scans are split into XY tiles that fit an 8 GB GPU; each
    tile stores orig_index.npy/core.npy so merge_tiles.py can stitch predictions.
    `meta` (dict of arrays) is saved to <out_dir>/../meta/<name>/ for later use.
    """
    out_dir = Path(out_dir)
    coord = (coord - coord.mean(0)).astype(np.float32)  # float32 needs local coords
    if color is None:
        print("  warning: no colour, filling zeros (colour helps a lot on heritage data)")
        color = np.zeros_like(coord, dtype=np.uint8)
    if normal is None:
        print("  estimating normals ...")
        normal = estimate_normals(coord)
    arrays = dict(coord=coord, color=np.asarray(color).astype(np.uint8),
                  normal=normal.astype(np.float32))
    if segment is not None:
        arrays["segment"] = np.asarray(segment).astype(np.int32)

    meta_dir = out_dir.parent / "meta" / name
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.save(meta_dir / "num_points.npy", np.array(len(coord)))
    for k, v in (meta or {}).items():
        np.save(meta_dir / f"{k}.npy", v)

    if not tile_size:
        _write(out_dir / name, arrays)
        return 1
    n = 0
    for n, (idx, core) in enumerate(tiles(coord, tile_size, buffer), 1):
        part = {k: v[idx] for k, v in arrays.items()}
        part["orig_index"], part["core"] = idx.astype(np.int64), core
        _write(out_dir / f"{name}__t{n - 1:03d}", part)
    return n


def _write(path, arrays):
    path.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(path / f"{k}.npy", v)
