# PointCloud-Mesh-Segmentation

Semantic segmentation of cultural heritage buildings (point clouds and meshes) **without training**:
renders + SAM 3 text prompts ("arch", "column", "vault", …) voted back onto 3D. Runs on an RTX 2080 (8 GB).

- **[docs/PLAN.md](docs/PLAN.md) — install and run (start here)**
- [docs/TRAINING.md](docs/TRAINING.md) — optional: fine-tune PTv3 + Sonata if you need higher accuracy
- [docs/RESEARCH.md](docs/RESEARCH.md) — survey of libraries and methods

```bash
bash envs/setup_zero_shot.sh && conda activate heritage-sam3
python scripts/zero_shot_segment.py building.las out/building --views both --save-views
```
