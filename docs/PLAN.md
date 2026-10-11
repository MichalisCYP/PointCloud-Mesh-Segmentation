# Plan — Point Cloud & Mesh Segmentation Workbench

Goal: get a set of segmentation methods running on **your own data** behind one interface, so you can compare them and pick what works. The survey behind this plan is in [RESEARCH.md](RESEARCH.md).

## Open questions (they decide which tracks matter)
1. **Data type.** Indoor scans (RGB-D/LiDAR), outdoor/aerial LiDAR, single objects/CAD meshes, or photogrammetry meshes?
2. **Task.** Semantic labels (wall/floor/car…), instances (each chair), or parts (handle/lid/leg)? Fixed classes or open-vocabulary/text queries?
3. **Labels.** Do you have annotated data for fine-tuning, or does it need to be zero-shot?
4. **Hardware.** GPU model and VRAM, or CPU-only?
5. **Licensing.** Research only, or commercial use?

## Proposed repo layout
```
segkit/
  io.py              # load/save .ply/.pcd/.las/.laz/.obj/.glb → common (points, colors, normals, faces) structure
  viz.py             # polyscope / open3d / rerun visualization of labels
  methods/
    base.py          # Segmenter interface: segment(data, **prompts) -> per-point/per-face labels
    classical.py     # Open3D RANSAC planes, DBSCAN, region growing; CGAL SDF for meshes
    sonata.py        # Sonata/PTv3 features + linear head (Pointcept)
    sam2point.py     # promptable zero-shot
    partfield.py     # mesh/object part segmentation
    p3sam.py         # Hunyuan3D-Part P3-SAM
  eval.py            # mIoU / AP / part-mIoU
envs/                # one conda env file or Dockerfile per heavy method
scripts/             # CLI: segment, compare, benchmark
data/samples/        # small test assets (git-lfs or download script)
```
Every method runs in **its own environment**, because CUDA dependencies conflict. A thin CLI calls each one and writes labels to a common format, such as `.npy` per-point labels or a PLY with a `label` field.

## Phases

**Phase 1 — Foundations & classical baselines (CPU, ~1–2 days)**
- `io.py` + `viz.py` with Open3D, trimesh, laspy, and polyscope
- Classical segmenters: RANSAC multi-plane extraction, DBSCAN/Euclidean clustering, normal-based region growing, mesh connected components, and CGAL SDF part segmentation
- Download sample data: a ScanNet/S3DIS room, a SemanticKITTI scan, and a few Objaverse/ShapeNet meshes

**Phase 2 — Learned scene segmentation (GPU)**
- Install the Sonata standalone package and run pretrained features plus the provided segmentation heads on sample scenes
- Wrap Pointcept PTv3 inference with released checkpoints (ScanNet / nuScenes)
- Optional: fine-tune on your labeled data (Sonata linear probe → full fine-tune)

**Phase 3 — Mesh / object part segmentation (GPU)**
- PartField: hierarchical, class-agnostic parts on meshes and point clouds
- P3-SAM: automatic and point-prompted part segmentation
- Compare both on PartObjaverse-Tiny and on your meshes; add SAMPart3D/SAMesh only if needed

**Phase 4 — Zero-shot / promptable / open-vocabulary**
- SAM2Point and Point-SAM for click-to-segment on raw point clouds
- OpenMask3D / Open3DIS if you have posed RGB-D frames and want text queries ("find the fire extinguisher")

**Phase 5 — Evaluation & comparison**
- `eval.py` with mIoU / AP / part-mIoU, plus a runtime and VRAM log per method
- One results table and screenshot gallery per dataset → choose a production method

## Suggested next step
Answer the open questions above. Phase 1 can start right away because it needs no GPU and is useful whatever the answers are.
