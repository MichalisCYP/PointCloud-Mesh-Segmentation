"""Zero-shot semantic segmentation of heritage buildings — no training.

Renders the point cloud (or a sampled mesh) from many virtual cameras, asks
SAM 3 to find each class by text ("column", "arch", "vaulted ceiling", ...),
and votes the 2D masks back onto the 3D points through a z-buffer.

Examples:
    # point cloud (PLY / LAS / LAZ / TXT), cameras orbiting the outside
    python zero_shot_segment.py church.las out/church --views orbit

    # interior scan: cameras inside looking around and up at vaults
    python zero_shot_segment.py nave.ply out/nave --views inside

    # textured mesh (OBJ/PLY/GLB), photogrammetry meshes are often Y-up
    python zero_shot_segment.py facade.obj out/facade --up y

    # check the renders first, without loading SAM 3
    python zero_shot_segment.py church.las out/church --render-only

Outputs (in the output folder):
    points_labelled.ply   points coloured by class, with a "label" property
    labels.npy            one label per input point (-1 = never seen)
    mesh_labelled.ply     (mesh input) faces coloured by class, per-face "label"
    views/                rendered views + mask overlays (with --save-views or --render-only)
    classes.json          label id -> class name
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from labels_to_mesh import colors_for, face_labels, write_labelled_mesh

# ArCH heritage taxonomy (ids 0-9). Each class: list of text prompts + weight.
# The weight down-ranks big generic surfaces so a column in front of a wall wins.
DEFAULT_CLASSES = [
    {"name": "arch", "prompts": ["arch"], "weight": 1.0},
    {"name": "column", "prompts": ["column", "pillar"], "weight": 1.0},
    {"name": "moldings", "prompts": ["molding", "cornice"], "weight": 1.0},
    {"name": "floor", "prompts": ["floor"], "weight": 0.8},
    {"name": "door_window", "prompts": ["door", "window"], "weight": 1.0},
    {"name": "wall", "prompts": ["wall"], "weight": 0.6},
    {"name": "stairs", "prompts": ["stairs"], "weight": 1.0},
    {"name": "vault", "prompts": ["vaulted ceiling", "dome"], "weight": 0.9},
    {"name": "roof", "prompts": ["roof"], "weight": 0.8},
    {"name": "other", "prompts": [], "weight": 1.0},  # seen but matched nothing
]


# ----------------------------------------------------------------------------- input
def load_input(path, up, max_points, mesh_points):
    """Return xyz (N,3) float64, rgb (N,3) uint8, mesh-or-None, face_index-or-None."""
    path = Path(path)
    ext = path.suffix.lower()
    mesh = face_index = None
    if ext in {".las", ".laz", ".txt", ".csv"} or (ext == ".ply" and not _ply_has_faces(path)):
        from prepare_pointcloud import read_las, read_ply, read_txt

        reader = {".las": read_las, ".laz": read_las, ".txt": read_txt, ".csv": read_txt}.get(ext, read_ply)
        xyz, rgb, _, _ = reader(path, None, "x,y,z,r,g,b")
    else:
        import trimesh

        mesh = trimesh.load(path, force="mesh", process=False)
        try:
            xyz, face_index, rgb = trimesh.sample.sample_surface(mesh, mesh_points, sample_color=True)
            rgb = np.asarray(rgb)[:, :3]
        except Exception:
            xyz, face_index = trimesh.sample.sample_surface(mesh, mesh_points)
            rgb = None
    if rgb is None:
        print("warning: no colour found; rendering grey shaded points (SAM 3 works much better with colour)")
        rgb = np.full((len(xyz), 3), 180, np.uint8)
    return np.asarray(xyz, np.float64), np.asarray(rgb).astype(np.uint8), mesh, face_index


def to_z_up(xyz, up):
    """Rendering assumes Z-up; outputs keep the original coordinates."""
    return np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], 1) if up == "y" else xyz


def _ply_has_faces(path):
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    return "face" in ply and ply["face"].count > 0


# ----------------------------------------------------------------------------- cameras
def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye
    f = f / np.linalg.norm(f)
    up = np.asarray(up, float)
    if abs(f @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, np.stack([r, u, f])  # rows: right, up, forward


def make_cameras(xyz, mode, fov_deg, n_azimuth, eye_height):
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    center = (lo + hi) / 2
    cams = []
    if mode in ("orbit", "both"):
        fov = math.radians(fov_deg)
        radius = 0.5 * np.linalg.norm(hi - lo) / math.sin(fov / 2) * 1.05
        for elev in (5, 25, 50):
            for k in range(n_azimuth):
                a, e = 2 * math.pi * k / n_azimuth, math.radians(elev)
                eye = center + radius * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
                cams.append((*look_at(eye, center), fov_deg))
    if mode in ("inside", "both"):
        eye = np.array([center[0], center[1], lo[2] + eye_height])
        for pitch in (0, 35, 70):
            for k in range(n_azimuth):
                a, p = 2 * math.pi * k / n_azimuth, math.radians(pitch)
                d = np.array([math.cos(a) * math.cos(p), math.sin(a) * math.cos(p), math.sin(p)])
                cams.append((*look_at(eye, eye + d), 90.0))
        cams.append((*look_at(eye, eye + np.array([0, 0, 1.0])), 90.0))  # straight up
    return cams


# ----------------------------------------------------------------------------- renderer
@torch.no_grad()
def render(xyz, rgb, cam, size, spacing, max_splat=6, bg=255):
    """Z-buffered point splatting. Returns image (H,W,3) uint8 and index map (H,W) int64 (-1 = empty)."""
    eye, R, fov_deg = cam
    W = H = size
    dev = xyz.device
    rel = (xyz - torch.as_tensor(eye, device=dev, dtype=xyz.dtype)) @ torch.as_tensor(R.T, device=dev, dtype=xyz.dtype)
    z = rel[:, 2]
    focal = (W / 2) / math.tan(math.radians(fov_deg) / 2)
    vis = z > 0.05
    idx = torch.nonzero(vis).squeeze(1)
    z = z[vis]
    px = (focal * rel[vis, 0] / z + W / 2).round().long()
    py = (-focal * rel[vis, 1] / z + H / 2).round().long()
    # splat radius ~ projected point spacing, so surfaces render closed
    rad = (focal * spacing / z / 2).ceil().clamp(0, max_splat).long()

    zbuf = torch.full((H * W,), float("inf"), device=dev, dtype=z.dtype)
    offsets = [(dx, dy) for dx in range(-max_splat, max_splat + 1) for dy in range(-max_splat, max_splat + 1)]
    for pass_ in (0, 1):
        idmap = torch.full((H * W,), -1, device=dev, dtype=torch.long)
        for dx, dy in offsets:
            m = rad >= max(abs(dx), abs(dy))
            x, y = px + dx, py + dy
            m &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
            pix = (y * W + x)[m]
            if pass_ == 0:
                zbuf.scatter_reduce_(0, pix, z[m], reduce="amin")
            else:
                front = z[m] <= zbuf[pix] * 1.002
                idmap[pix[front]] = idx[m][front]
    img = torch.full((H * W, 3), bg, device=dev, dtype=torch.uint8)
    hit = idmap >= 0
    img[hit] = rgb[idmap[hit]]
    return img.view(H, W, 3).cpu().numpy(), idmap.view(H, W)


# ----------------------------------------------------------------------------- SAM 3
def load_sam3(threshold):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model()
    return Sam3Processor(model, confidence_threshold=threshold)


def class_scores(processor, image, classes):
    """(C, H, W) best score per pixel and class for one rendered view."""
    from PIL import Image

    state = processor.set_image(Image.fromarray(image))
    H, W = image.shape[:2]
    out = torch.zeros((len(classes), H, W), device="cuda" if torch.cuda.is_available() else "cpu")
    for c, cls in enumerate(classes):
        for prompt in cls["prompts"]:
            res = processor.set_text_prompt(prompt=prompt, state=state)
            if len(res["scores"]) == 0:
                continue
            masks = res["masks"].squeeze(1).float()                     # (K,H,W)
            s = (masks * res["scores"].float().view(-1, 1, 1)).amax(0)  # (H,W)
            out[c] = torch.maximum(out[c], s.to(out.device) * cls.get("weight", 1.0))
        processor.reset_all_prompts(state)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="point cloud (.ply/.las/.laz/.txt) or mesh (.obj/.ply/.glb)")
    ap.add_argument("out_dir")
    ap.add_argument("--views", choices=["orbit", "inside", "both"], default="orbit")
    ap.add_argument("--azimuths", type=int, default=12, help="camera directions per ring")
    ap.add_argument("--fov", type=float, default=50.0, help="orbit camera field of view (deg)")
    ap.add_argument("--eye-height", type=float, default=1.7, help="inside cameras: metres above floor")
    ap.add_argument("--size", type=int, default=1008, help="render resolution (SAM 3 native: 1008)")
    ap.add_argument("--up", choices=["z", "y"], default="z")
    ap.add_argument("--max-points", type=int, default=6_000_000, help="points rendered; the rest get kNN labels")
    ap.add_argument("--mesh-points", type=int, default=4_000_000, help="points sampled from a mesh")
    ap.add_argument("--classes", default=None, help="JSON file with a class/prompt list (default: ArCH)")
    ap.add_argument("--threshold", type=float, default=0.4, help="SAM 3 detection confidence")
    ap.add_argument("--min-score", type=float, default=0.3, help="below this a seen point becomes 'other'")
    ap.add_argument("--save-views", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="only write the renders, skip SAM 3")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "views").mkdir(parents=True, exist_ok=True)
    classes = json.load(open(args.classes)) if args.classes else DEFAULT_CLASSES
    other = next((i for i, c in enumerate(classes) if not c["prompts"]), None)
    json.dump({i: c["name"] for i, c in enumerate(classes)}, open(out / "classes.json", "w"), indent=1)

    xyz_all, rgb_all, mesh, face_index = load_input(args.input, args.up, args.max_points, args.mesh_points)
    n_all = len(xyz_all)
    sub = np.arange(n_all)
    if n_all > args.max_points:
        sub = np.sort(np.random.default_rng(0).choice(n_all, args.max_points, replace=False))
    print(f"{args.input}: {n_all} points, rendering {len(sub)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xyz_r = to_z_up(xyz_all[sub], args.up)
    xyz = torch.as_tensor(xyz_r - xyz_r.mean(0), dtype=torch.float32, device=dev)
    rgb = torch.as_tensor(rgb_all[sub], device=dev)
    probe = xyz[torch.randperm(len(xyz), device=dev)[:20000]].cpu().numpy()
    spacing = float(np.median(cKDTree(xyz.cpu().numpy()).query(probe, k=2)[0][:, 1]))
    cams = make_cameras(xyz.cpu().numpy(), args.views, args.fov, args.azimuths, args.eye_height)
    print(f"point spacing ~{spacing * 100:.1f} cm, {len(cams)} views ({args.views})")

    processor = None if args.render_only else load_sam3(args.threshold)
    votes = torch.zeros((len(sub), len(classes)), device=dev)
    seen = torch.zeros(len(sub), device=dev)
    for v, cam in enumerate(cams):
        img, idmap = render(xyz, rgb, cam, args.size, spacing)
        if args.render_only or args.save_views:
            from PIL import Image

            Image.fromarray(img).save(out / "views" / f"view_{v:03d}.jpg", quality=90)
        if args.render_only:
            continue
        scores = class_scores(processor, img, classes).to(dev)        # (C,H,W)
        hit = idmap >= 0
        pts = idmap[hit]
        votes.index_put_((pts,), scores[:, hit].T, accumulate=True)
        seen.index_put_((pts,), torch.ones_like(pts, dtype=seen.dtype), accumulate=True)
        if args.save_views:
            from PIL import Image

            best, lab = scores.max(0)
            ov = colors_for(np.where(best.cpu().numpy() > 0, lab.cpu().numpy(), -1))
            blend = np.where((best.cpu().numpy() > 0)[..., None], (0.5 * img + 0.5 * ov).astype(np.uint8), img)
            Image.fromarray(blend).save(out / "views" / f"view_{v:03d}_labels.jpg", quality=90)
        print(f"  view {v + 1}/{len(cams)}: {int(hit.sum())} px on surface")

    if args.render_only:
        print(f"renders written to {out / 'views'}")
        return

    # per-point label: highest mean score; seen-but-weak -> 'other'; unseen -> -1 (filled below)
    mean = votes / seen.clamp(min=1).unsqueeze(1)
    best, label = mean.max(1)
    label = label.cpu().numpy()
    best, seen = best.cpu().numpy(), seen.cpu().numpy()
    if other is not None:
        label[(seen > 0) & (best < args.min_score)] = other
    label[seen == 0] = -1

    # propagate to points that were never visible / not rendered
    labels = np.full(n_all, -1, np.int64)
    labels[sub] = label
    known = labels >= 0
    if known.any() and (~known).any():
        _, nn = cKDTree(xyz_all[known]).query(xyz_all[~known], workers=-1)
        labels[~known] = labels[known][nn]
    np.save(out / "labels.npy", labels)

    from plyfile import PlyData, PlyElement

    pts = np.empty(n_all, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("red", "u1"), ("green", "u1"),
                                 ("blue", "u1"), ("label", "i4")])
    pts["x"], pts["y"], pts["z"] = xyz_all.T
    pts["red"], pts["green"], pts["blue"] = colors_for(labels).T
    pts["label"] = labels
    PlyData([PlyElement.describe(pts, "vertex")]).write(out / "points_labelled.ply")

    if mesh is not None:
        flab, _ = face_labels(mesh, face_index, labels)
        write_labelled_mesh(mesh, flab, out / "mesh_labelled.ply")

    counts = np.bincount(labels[labels >= 0], minlength=len(classes))
    print("class counts:", {c["name"]: int(n) for c, n in zip(classes, counts)})
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
