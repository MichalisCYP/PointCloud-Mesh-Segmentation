# Plan — Heritage Building Segmentation, No Training

Runs on your GPU PC (RTX 2080, 8 GB). **No training and no labelled data needed.**

## How it works
1. Render your point cloud or mesh from many virtual cameras (outside orbit and/or inside the building).
2. **SAM 3** (Meta, 2025) finds each class in every render from a text prompt: "arch", "column", "molding", "vaulted ceiling", …
3. Vote each 2D mask back onto the 3D points it shows, keeping the best class per point.
4. Points that are never visible copy the label of their nearest neighbour. Meshes get per-face labels by voting.

**Default classes** use the ArCH heritage taxonomy (label IDs 0–9): arch, column, moldings, floor, door/window, wall, stairs, vault, roof, other. You can change them; see below.

---

## Step 1 — Install (once, on the GPU PC)

1. **Get access to the SAM 3 weights.** Request access at <https://huggingface.co/facebook/sam3> and wait for approval. Then create a read token on Hugging Face and run `hf auth login` (or `huggingface-cli login`).
2. **Run the setup:**
   ```bash
   git clone https://github.com/michaliscyp/pointcloud-mesh-segmentation.git
   cd pointcloud-mesh-segmentation
   bash envs/setup_zero_shot.sh
   conda activate heritage-sam3
   ```

The setup ends with a smoke test that should print the peak VRAM and `OK`.

Requirements: Linux or **WSL2 on Windows**, NVIDIA driver ≥ 570, conda.

## Step 2 — Run

```bash
# Exterior of a building (point cloud: .ply / .las / .laz / .txt)
python scripts/zero_shot_segment.py church.las out/church --views orbit --save-views

# Interior scan: cameras stand inside and look around and up at the vaults
python scripts/zero_shot_segment.py nave.ply out/nave --views inside --save-views

# Both
python scripts/zero_shot_segment.py chapel.laz out/chapel --views both

# Textured mesh (photogrammetry exports are usually Y-up)
python scripts/zero_shot_segment.py facade.obj out/facade --up y --save-views
```

**Outputs** in `out/<name>/`:

| File | Content |
|---|---|
| `points_labelled.ply` | Points coloured by class, plus a `label` field. Open in CloudCompare. |
| `mesh_labelled.ply` | Meshes only: faces coloured by class, with a per-face `label`. Open in MeshLab or CloudCompare. |
| `labels.npy` | One label per point |
| `classes.json` | Label ID → class name |
| `views/` | Renders and label overlays (with `--save-views`) |

**Tip:** run once with `--render-only` first. It is fast and doesn't need SAM 3. Look at `views/` and check that the building is in frame, upright, and dense enough. If it looks upside down or sideways, add `--up y`.

## Step 3 — Tune (no training, just settings)

| Problem | Fix |
|---|---|
| Elements missed | Lower `--threshold` (0.4 → 0.3). Add prompt synonyms in a classes file. |
| False detections | Raise `--threshold` (→ 0.5) |
| Walls swallowing columns or moldings | Lower the `weight` of "wall" in the classes file |
| Parts never seen (courtyards, behind columns) | `--views both`, more `--azimuths` (e.g. 16) |
| Gaps and holes in the renders | The scan is sparse for the image size. Lower `--size` (e.g. 768) so the points cover the pixels. |
| Out of GPU memory | `--size 768`. Lower `--max-points`. |
| Slow | Fewer `--azimuths`. The run time is mostly SAM 3 (views × prompts). |

**Custom classes:** write a JSON file and pass `--classes my_classes.json`. Order defines the label IDs. A class with an empty prompt list is the fallback for "seen but nothing matched".
```json
[
  {"name": "column",  "prompts": ["column", "pillar", "pilaster"], "weight": 1.0},
  {"name": "capital", "prompts": ["column capital"],                "weight": 1.2},
  {"name": "window",  "prompts": ["window", "stained glass window"], "weight": 1.0},
  {"name": "wall",    "prompts": ["stone wall"],                    "weight": 0.6},
  {"name": "other",   "prompts": [],                                "weight": 1.0}
]
```

## What to expect
- **Strongest:** clearly visible elements with colour, such as columns, arches, windows/doors, roofs, floors, and stairs.
- **Weaker:** subtle geometry such as thin mouldings, and points that are hard to see from any camera.
- **Colour matters:** scans without RGB render grey, and SAM 3 does much worse on them.
- **For publishable accuracy:** if zero-shot isn't good enough, the training route (fine-tuning Point Transformer V3 + Sonata) is documented in [TRAINING.md](TRAINING.md). The zero-shot labels can be corrected in CloudCompare and used as its training data.

## Files
| Path | Purpose |
|---|---|
| `envs/setup_zero_shot.sh` | Conda env + SAM 3 + smoke test |
| `scripts/zero_shot_segment.py` | The whole no-training pipeline |
| `scripts/labels_to_mesh.py` | Point labels → mesh face labels (used internally, also standalone) |
| `docs/TRAINING.md`, `envs/setup_2080.sh`, `pointcept_configs/` | Optional training route |
