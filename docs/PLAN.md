# Plan — Segmentation for Cultural Heritage Buildings

Two tracks: **point clouds** and **meshes**. Both use the same model, so you only train once.
The full survey is in [RESEARCH.md](RESEARCH.md).

## The algorithm: Point Transformer V3 + Sonata pre-training (Pointcept)

- **What it is:** PTv3 (CVPR'24) is the strongest open 3D segmentation backbone. Sonata (CVPR'25) is a self-supervised pre-trained PTv3 that you fine-tune on your labels.
- **Heritage evidence:** in a 2025 heritage study on ArCH, Point Transformer v1 generalised best of five models (61.3 mIoU). PTv3 is that model's faster, stronger successor.
- **Heritage training data:** the **ArCH** benchmark ([archdataset.polito.it](https://archdataset.polito.it)) is an annotated heritage point cloud set. Its classes are arch, column, moldings, floor, door/window, wall, stairs, vault, roof, and other.
- **Label set:** use the ArCH classes for your own labels, so you can train on ArCH + your data together.

---

## Track A — Point cloud segmentation

1. **Install** [Pointcept](https://github.com/Pointcept/Pointcept). On the RTX 5060 you need PyTorch ≥ 2.7 built with CUDA 12.8. If `spconv` / `flash-attn` won't install, compile them with `TORCH_CUDA_ARCH_LIST="12.0"`, or set `enable_flash=False`.
2. **Convert** each scan:
   ```
   python scripts/prepare_pointcloud.py scan.las data/heritage/train/scan01 --label-field classification
   ```
   Put labelled scans in `train/` and `val/`, unlabelled ones in `test/`. Download ArCH and convert it the same way into `train/`.
3. **Train:** copy a Sonata fine-tune config from Pointcept's `configs/` (the ScanNet semseg Sonata one). Change the dataset path, `num_classes`, and the class names. Then run:
   ```
   sh scripts/train.sh -g 1 -d heritage -c semseg-sonata-heritage -n run1
   ```
   - **8 GB card:** batch size 1–2, grid size 2–3 cm, AMP on.
   - **If it runs out of memory:** freeze the backbone (decoder-probe config) or train on a bigger GPU.
4. **Predict:** `sh scripts/test.sh ...` writes `<scene>_pred.npy` per scan.

**No labels at all for a building?** Use [Point-SAM](https://github.com/zyc00/Point-SAM) or [SAM2Point](https://github.com/ZiyuGuo99/SAM2Point) to click-segment elements. Use the result to label a few scans quickly, then fine-tune as above.

---

## Track B — Mesh segmentation (photogrammetry / textured meshes)

**The best working approach is: sample the mesh → run the same PTv3 model → vote labels back onto the faces.** Mesh-native networks are weaker and have no heritage training data.

```
python scripts/mesh_to_pointcloud.py building.obj data/heritage/test/building01 --points 2000000
# run Track A prediction on it
python scripts/labels_to_mesh.py building.obj data/heritage/test/building01/face_index.npy \
       exp/heritage/run1/result/building01_pred.npy building01_segmented.ply
```
- Texture colour is sampled into the points, which helps with mouldings and openings.
- Labelled meshes (`.ply` with a per-face `label`) give `segment.npy` and can be used for training.

**Splitting into parts without class labels** (e.g. separating each column or capital): run [PartField](https://research.nvidia.com/labs/toronto-ai/partfield-release) (NVIDIA) on a single element or small structure, cropped out of the building first. It gives hierarchical parts with no training.

---

## Order of work
1. Install Pointcept and run the pretrained Sonata demo once, to prove the GPU setup works.
2. Convert ArCH + your labelled scans → fine-tune → check val mIoU.
3. Predict on unlabelled point clouds.
4. Run meshes through `mesh_to_pointcloud.py` → predict → `labels_to_mesh.py`.
