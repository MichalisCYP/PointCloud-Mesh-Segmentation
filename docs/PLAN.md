# Plan — Point Cloud & Mesh Segmentation Workbench

Goal: get a set of segmentation methods running on **your own data** behind one interface, so you can compare them and pick what works. The survey behind this plan is in [RESEARCH.md](RESEARCH.md).

## Your setup (answered 2026-10)
| Question | Answer | What it means for the plan |
|---|---|---|
| Data | Indoor and outdoor scans, kept as **separate** datasets | Two tracks. Each gets its own model, config, and class set; don't mix domains in one fine-tune. |
| Task / labels | Some scans are labelled, many aren't | **Semi-supervised**: fine-tune a pretrained backbone on the labelled scans, use zero-shot / clustering on the rest, then optionally pseudo-label. |
| GPUs | RTX 5060 8 GB + others | 8 GB for development and inference; the bigger cards for training (see the 8 GB notes below). |
| Use | Research | Non-commercial research licenses (NVIDIA, CC-BY-NC weights, etc.) are fine. |

Still useful to know: the label classes and file format, the sensor (TLS / MLS / ALS / RGB-D), typical points per scan, whether there's RGB, and which the "other" GPUs are.

### Notes for the RTX 5060 (8 GB, Blackwell)
- Blackwell needs **PyTorch built for CUDA 12.8 or newer** (sm_120). Older wheels of `spconv`, `torch-scatter`, and `flash-attn` may not have sm_120 kernels. Check each wheel first and be ready to build from source. This is the most likely install blocker.
- PTv3 / Sonata can run **without flash-attn** (`enable_flash=False`), at the cost of more memory and lower speed.
- Fitting into 8 GB:
  - Use grid sampling (indoor 2–5 cm, outdoor 5–10 cm).
  - Run inference on overlapping chunks or spheres.
  - Use AMP (fp16/bf16) and batch size 1–2.
  - For training on 8 GB, freeze the backbone and train only the head (linear probe or decoder). Do full fine-tunes on the larger GPUs.

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

## Phases (re-prioritised for indoor + outdoor scans with partial labels)

**Phase 1 — Foundations & classical baselines (CPU)**
- `io.py` + `viz.py`: PLY / PCD / LAS / LAZ / E57 loading with laspy, Open3D, and pye57, normalised to a common structure plus a label array. Visualize with polyscope or rerun.
- Convert both of your labelled datasets to **Pointcept's format** (per-scene `coord.npy`, `color.npy`, `normal.npy`, `segment.npy`), with a fixed train/val/test split by scene.
- Classical baselines:
  - **Indoor:** RANSAC multi-plane extraction (floor, ceiling, walls) and DBSCAN on what's left.
  - **Outdoor:** CSF / SMRF ground filtering (PDAL), then Euclidean clustering of objects.
- These baselines set a floor and also serve as cheap pseudo-labels.

**Phase 2 — Learned semantic segmentation, one track per domain (GPU)** ← main work
1. **Environment:** Pointcept + Sonata on the 5060, with a CUDA 12.8+ PyTorch. Smoke-test pretrained inference on ScanNet (indoor) and nuScenes / SemanticKITTI (outdoor) samples.
2. **Linear probe** a frozen Sonata (or Concerto) backbone on your labelled scans. This fits in 8 GB and is the quickest way to see how far pretrained features get you.
3. **Fine-tune** PTv3 initialised from Sonata on the bigger GPUs. Compare against training from scratch and against a lighter backbone (SpUNet / MinkUNet) as a sanity check.
4. Optional: **pre-train with self-supervision on your unlabelled scans** (Sonata-style), then fine-tune. This is how the unlabelled data adds accuracy. It needs the large GPUs.

**Phase 3 — Using the unlabelled scans**
- **Pseudo-labelling / self-training:**
  1. Predict with the Phase 2 model.
  2. Keep high-confidence points.
  3. Retrain.
  4. Measure whether this beats Phase 2.
- **Zero-shot / promptable tools** for exploring and speeding up annotation: SAM2Point and Point-SAM (click-to-segment), and Open3DIS / OpenMask3D if any indoor scans have posed RGB frames (text queries).
- Optional **instance segmentation**, if you need individual objects: OneFormer3D (indoor) or PTv3 + PointGroup in Pointcept.

**Phase 4 — Evaluation & reporting**
- `eval.py`: mIoU, per-class IoU, overall accuracy (plus AP if instances are needed), and runtime and peak VRAM per method.
- One results table per domain covering classical, linear probe, fine-tune, SSL + fine-tune, and pseudo-label.

**Deprioritised:** mesh part segmentation (PartField, P3-SAM). Scans are scene-level, so this only comes back if you need object parts.

## Suggested next step
Start Phase 1: the data loaders and the Pointcept-format converter. To write the converters, I need one sample scan from each domain, or a description of the file format and label fields.
