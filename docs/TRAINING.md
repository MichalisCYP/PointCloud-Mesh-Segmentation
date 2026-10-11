# Optional — Training Route (PTv3 + Sonata on an RTX 2080)

_Only needed if the no-training pipeline in [PLAN.md](PLAN.md) is not accurate enough._

**Model:** Point Transformer V3 with **Sonata** pre-trained weights (Pointcept), fine-tuned on your labels.
**Classes:** ArCH heritage taxonomy — arch, column, moldings, floor, door/window, wall, stairs, vault, roof, other.
**Point clouds** go straight into the model. **Meshes** are sampled into points, segmented, and the labels voted back onto the faces.

Survey and alternatives: [RESEARCH.md](RESEARCH.md).

---

## Step 1 — Install (once, ~30 min)

```bash
bash envs/setup_2080.sh            # creates conda env "pointcept-2080", clones Pointcept next to this repo
conda activate pointcept-2080
cd ../Pointcept
```
The script does the following:
- Installs PyTorch 2.5 + CUDA 12.4 (supports the 2080) and **skips FlashAttention**, which doesn't support RTX 20xx.
- Builds the CUDA ops for sm_75.
- Downloads the Sonata weights.
- Links `configs/heritage` and `data/heritage` from this repo.
- Finishes with a **GPU smoke test** that should print `OK` and the peak VRAM.

If the smoke test runs out of memory, retry it with `--points 50000`, then set the same number as `point_max` in the configs.

Requirements: Linux or WSL2, NVIDIA driver ≥ 550, conda.

## Step 2 — Prepare data

Run from the **repo** folder (same env). Each command writes `data/heritage/<split>/`.

```bash
# Labelled point clouds  -> train (most) and val (1–2 buildings you never train on)
python scripts/prepare_pointcloud.py church.las   data/heritage/train --label-field classification --tile 10
python scripts/prepare_pointcloud.py cloister.ply data/heritage/val   --label-field scalar_label  --tile 10

# ArCH benchmark scenes (download from archdataset.polito.it) -> train
python scripts/prepare_pointcloud.py ArCH_scene.txt data/heritage/train --txt-cols x,y,z,r,g,b,label --tile 10

# Unlabelled point clouds and meshes to segment -> test
python scripts/prepare_pointcloud.py chapel.laz  data/heritage/test --tile 10
python scripts/mesh_to_pointcloud.py facade.obj  data/heritage/test --points 3000000 --tile 10
```
- **Labels must use the 10 ArCH class IDs (0–9).** Use `-1` for unlabelled points. Remap your IDs first if they differ.
- `--tile 10` splits buildings into 10 m tiles (plus a 2 m context border) so they fit in 8 GB.
- `--voxel 0.01` thins very dense TLS scans. The model works at 2 cm anyway.
- Normals are computed automatically when missing. Colour is strongly recommended; missing colour is filled with zeros.
- Check `--txt-cols` against ArCH's actual column order before converting.

## Step 3 — Train (in `../Pointcept`)

```bash
# A) Decoder probing: Sonata encoder frozen, fits comfortably in 8 GB. Start here.
sh scripts/train.sh -g 1 -d heritage -c semseg-sonata-heritage-dec -n dec \
   -w weights/sonata/pretrain-sonata-v1m1-0-base.pth

# B) Full fine-tuning: better accuracy, tight on 8 GB (40k-point crops). Try after A works.
sh scripts/train.sh -g 1 -d heritage -c semseg-sonata-heritage-ft -n ft \
   -w weights/sonata/pretrain-sonata-v1m1-0-base.pth
```
- Validation mIoU is logged every `eval_epoch`, and the best model is saved to `exp/heritage/<name>/model/model_best.pth`.
- **If you run out of memory:** lower `point_max` in the config's `SphereCrop` (80000 → 50000 for dec; 40000 → 25000 for ft).
- **Training time:** with a few hundred tiles, expect roughly a day for A on a 2080. Lower `epoch` for a quick first run.

## Step 4 — Predict

```bash
# in ../Pointcept: runs on data/heritage/test, writes exp/heritage/dec/result/<tile>_pred.npy
sh scripts/test.sh -g 1 -d heritage -n dec -w model_best

# back in the repo: stitch tiles into one prediction per building
python scripts/merge_tiles.py data/heritage/test ../Pointcept/exp/heritage/dec/result
#   -> ../Pointcept/exp/heritage/dec/result/merged/<scene>_pred.npy  (one label per prepared point)

# meshes: put labels back on the faces -> coloured .ply with a per-face "label"
python scripts/labels_to_mesh.py facade.obj data/heritage/meta/facade/face_index.npy \
       ../Pointcept/exp/heritage/dec/result/merged/facade_pred.npy facade_segmented.ply
```
Open the results in CloudCompare or MeshLab.

## Step 5 — Improve (optional)

| If… | Do |
|---|---|
| A works and you have time | Run B (full fine-tune) and compare val mIoU |
| Too few labelled buildings | Pre-train on ArCH first, then fine-tune on yours (`-w` = the ArCH-trained `model_best.pth`) |
| Need more labels quickly | Predict, fix the mistakes in CloudCompare, add the result to `train/`, retrain |
| Want to split one element into parts without labels (e.g. a capital) | [PartField](https://research.nvidia.com/labs/toronto-ai/partfield-release) on the cropped mesh |
| Get access to a bigger GPU | Same configs; raise `batch_size` and `point_max` |

---

### Files in this repo
| Path | Purpose |
|---|---|
| `envs/setup_2080.sh` | Environment + weights + smoke test |
| `pointcept_configs/heritage/semseg-sonata-heritage-{dec,ft}.py` | 2080-tuned Pointcept configs (no flash attention, fp16, batch 2) |
| `scripts/prepare_pointcloud.py` | PLY/LAS/LAZ/TXT → Pointcept format (normals, tiling) |
| `scripts/mesh_to_pointcloud.py` | Mesh → sampled points (+ texture colour, face index) |
| `scripts/merge_tiles.py` | Tile predictions → one prediction per building |
| `scripts/labels_to_mesh.py` | Point predictions → per-face mesh labels |
| `scripts/gpu_smoke_test.py` | One training step on random data, reports peak VRAM |
