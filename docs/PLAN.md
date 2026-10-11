# Plan — Semantic Segmentation of Indoor & Outdoor Scans

**Goal (research):** find the most accurate semantic segmentation for two separate scan domains (indoor and outdoor). Each domain has some labelled and some unlabelled scans. Compare classical, pretrained-probe, fine-tuned, and semi-supervised approaches **under one evaluation protocol**.

The survey behind this plan is in [RESEARCH.md](RESEARCH.md).

---

## 1. Constraints & decisions

| Item | Value | Decision |
|---|---|---|
| Domains | Indoor and outdoor, separate datasets | **Two independent tracks.** Separate class sets, configs, and checkpoints. No cross-domain mixing until the per-domain baselines are done. |
| Labels | Partial | Labelled → train/val/test. Unlabelled → self-supervised pre-training and pseudo-labelling. |
| GPUs | RTX 5060 8 GB (dev/inference) + larger GPUs (training) | 5060: preprocessing, inference, linear/decoder probes. Big GPUs: full fine-tunes, SSL pre-training. |
| Use | Research | Non-commercial licenses are acceptable. Record the license of every model and checkpoint used, for the paper. |
| Core framework | **Pointcept** (PTv3 / Sonata / Concerto) | One codebase for all learned methods. Results stay comparable and configs reproducible. |

### Still needed from you (blocks Milestone 1)
- [ ] File format(s) and how labels are stored (LAS classification field, PLY scalar field, separate files…)
- [ ] Class list per domain, plus an "ignore/unlabelled" value
- [ ] Sensor per domain (TLS / MLS / ALS / RGB-D), whether there's RGB and intensity, and typical points per scan
- [ ] Number of labelled vs unlabelled scans per domain
- [ ] Which the "other" GPUs are (model and VRAM)

---

## 2. Environment

**The 5060 (Blackwell, sm_120) cannot use the stock Sonata/Pointcept environments.** Sonata's standalone env pins CUDA 12.4 / PyTorch 2.5, which has no sm_120 kernels. Native sm_120 support arrived in PyTorch 2.7 with CUDA 12.8.

| Env | Where | Spec |
|---|---|---|
| `seg-core` | Any machine, CPU | Python 3.11, numpy, open3d, laspy[lazrs], pye57, pdal, polyscope / rerun, scikit-learn |
| `seg-pointcept-bw` | 5060 | Driver ≥ 570, **PyTorch ≥ 2.7 + cu128**. Build `spconv`, `torch-scatter`, Pointcept ops, and optionally `flash-attn` from source with `TORCH_CUDA_ARCH_LIST="12.0"` if no sm_120 wheel exists. |
| `seg-pointcept` | Big GPUs | Same package versions as `seg-pointcept-bw`, with the arch list for that GPU. Version parity keeps checkpoints interchangeable. |

- Ship each environment as a **Dockerfile** in `envs/`, plus a lockfile.
- **Smoke test:** run the Sonata ScanNet linear-probe checkpoint on a ScanNet scene and check the predictions look right. **If this fails on the 5060, fall back** to `enable_flash=False` first, then to running the GPU work on the big GPUs only.

---

## 3. Data conventions

- **Canonical on-disk format** is Pointcept's: `data/<domain>/<split>/<scene>/{coord,color,normal,strength,segment}.npy`. `segment` is `int`, with `-1` meaning ignore.
- **Splits** are by scene, never by points, and fixed by a seed in `data/<domain>/splits.json`. Hold out the test set once and don't tune on it.
- **Class map** goes in `configs/<domain>/classes.yaml`. Original label IDs map to contiguous training IDs.
- **Preprocessing:**
  - Center each scene.
  - Grid-subsample to a fixed voxel size (indoor ≈ 2 cm, outdoor ≈ 5–10 cm; tune once).
  - Compute normals if missing.
  - Large outdoor scans: tile into blocks of about 50 m with overlap.
- **Dataset stats report:** points per scene, class frequencies, and label coverage. This drives class weighting and decides whether rare classes need merging.

---

## 4. Milestones

Each milestone has a deliverable and an exit criterion. Don't start the next one until the criterion is met.

### M1 — Data pipeline & baselines (CPU) · ~1 week
- `segkit/io.py`: readers for your formats → canonical arrays. Round-trip tests.
- `segkit/convert.py`: raw → Pointcept format, splits, class map, stats report.
- `segkit/viz.py`: labels and predictions side by side, plus a per-class error view.
- `segkit/eval.py`: mIoU, per-class IoU, OA, and confusion matrix. Evaluated at **full resolution**, with predictions mapped back from the voxels to the original points.
- **Classical baselines:**
  - **Indoor:** RANSAC planes (floor, ceiling, walls) + DBSCAN.
  - **Outdoor:** CSF/SMRF ground filtering + clustering + height/geometric-feature random forest.
- **Exit:** both domains converted. The baseline mIoU table is produced on val, and visual checks pass.

### M2 — Environment & pretrained smoke test (5060) · ~3–5 days
- Build `seg-pointcept-bw` and reproduce the Sonata ScanNet linear-probe result on a few public scenes.
- Run Sonata features on your scans and look at PCA/t-SNE of the features against your labels, as a qualitative check.
- **Exit:** reproducible inference on the 5060, with peak VRAM recorded per scene size.

### M3 — Supervised learning on labelled data · ~2 weeks
The experiment matrix, run per domain:

| ID | Model | Init | Trainable | GPU |
|---|---|---|---|---|
| E1 | SpUNet / MinkUNet | scratch | all | big |
| E2 | PTv3 | scratch | all | big |
| E3 | PTv3 | Sonata | linear head | 5060 |
| E4 | PTv3 | Sonata | decoder | 5060 / big |
| E5 | PTv3 | Sonata | all (full FT) | big |
| E6 | PTv3 | Concerto (if weights fit the domain) | all | big |

- Run every experiment with 3 seeds; report mean ± std.
- **Label-efficiency curve:** repeat E2/E3/E5 with 10% / 25% / 50% / 100% of the labelled scenes. This is the key research figure.
- **Exit:** a best supervised model per domain is selected on val.

### M4 — Exploit unlabelled scans · ~2–3 weeks
- **E7 — domain SSL:** continue Sonata self-supervised pre-training on all your scans of that domain, then fine-tune as in E5.
- **E8 — self-training:**
  1. Pseudo-label the unlabelled scans with the best model, keeping per-class confidence-threshold points.
  2. Retrain.
  3. Repeat for 1–2 rounds.
- **E9 — E7 + E8 combined.**
- Optional annotation helper: SAM2Point or Point-SAM for click-based labelling of extra scenes.
- **Exit:** a decision on whether unlabelled data helps (E7–E9 vs the best of M3), with significance across seeds.

### M5 — Final evaluation & write-up · ~1 week
- Evaluate the chosen models once on the held-out **test** split.
- Measure robustness on scans that differ in density and sensor, as an ablation.
- Deliverables:
  - Results tables
  - The label-efficiency plot
  - A qualitative figure gallery
  - Runtime and VRAM per method
  - Configs and checkpoints tagged in git

---

## 5. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| spconv / flash-attn don't build for sm_120 | High | Build from source with arch 12.0. Disable flash. Do GPU work on the big GPUs only. |
| 8 GB OOM on large scenes | High | Larger voxels, sphere/chunk inference, AMP, batch 1. Train only probes on the 5060. |
| Outdoor domain far from Sonata's pre-training data | Medium | E7 (domain SSL) and the E1/E2 from-scratch baselines show it directly. |
| Too few labelled scenes or severe class imbalance | Medium | Class-weighted / Lovász loss, merge rare classes, label-efficiency curve. Use SAM2Point-assisted labelling. |
| Label leakage via overlapping tiles | Medium | Split by scene before tiling. Never tile across splits. |
| Results not reproducible | Low | Docker + lockfiles, fixed seeds, configs in git, 3 seeds per run. |

---

## 6. Repo layout
```
segkit/          io.py convert.py viz.py eval.py baselines/ (classical methods)
configs/         indoor/ outdoor/  → classes.yaml + Pointcept configs per experiment ID (E1…E9)
envs/            Dockerfile.core  Dockerfile.pointcept  (+ lockfiles)
scripts/         convert.sh  train.sh  eval.sh  report.py
data/            (git-ignored) raw/ processed/ splits.json
results/         per-experiment metrics JSON → aggregated tables/plots
docs/            RESEARCH.md  PLAN.md
```

**Out of scope for now:** mesh part segmentation (PartField, P3-SAM) and open-vocabulary methods (OpenMask3D, Open3DIS). They can be added later as extra experiments if the research question widens.
