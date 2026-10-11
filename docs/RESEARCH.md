# Point Cloud & Mesh Segmentation — Landscape Survey

_Last updated: 2026-10. Things change quickly in this field. Check each repo's README, license, and last commit before you rely on it._

## 0. TL;DR — recommended stack

| Need | First pick | Runner-up / alternative |
|---|---|---|
| Classical geometry ops (planes, clusters, normals, I/O) | **Open3D** (Python) | PCL (C++), CGAL (C++/Python bindings) |
| Mesh processing / I/O | **trimesh** + **PyMeshLab** | libigl, CGAL |
| Large aerial / terrestrial LiDAR (LAS/LAZ) | **PDAL** + **laspy** | CloudCompare (GUI + CLI) |
| Learned semantic segmentation of scenes (indoor / outdoor) | **Pointcept** → PTv3 / **Sonata** / Concerto | Open3D-ML (RandLA-Net, KPConv), MMDetection3D |
| 3D instance segmentation (scenes) | **OneFormer3D** / **Mask3D** | Pointcept (PTv3 + PointGroup), SPFormer |
| Zero-shot / promptable point cloud segmentation | **SAM2Point**, **Point-SAM** | SAI3D, Segment3D |
| Open-vocabulary (text-queried) scene segmentation | **Open3DIS**, **OpenMask3D** | OpenIns3D |
| Mesh / object **part** segmentation (class-agnostic) | **PartField** (NVIDIA), **P3-SAM** (Tencent Hunyuan3D-Part) | SAMPart3D, SAMesh, PartSAM |
| Mesh/point learning on arbitrary surfaces | **DiffusionNet** | MeshCNN (older) |
| Visualization | **Polyscope**, Open3D viewer, **rerun** | CloudCompare, MeshLab |

---

## 1. Classical / geometric (no training needed)

These are fast and easy to understand. They make good baselines and preprocessing steps, and they are usually enough for structured scenes such as planes, buildings, and industrial parts.

| Library | Lang | Segmentation-relevant features | Notes |
|---|---|---|---|
| [Open3D](https://www.open3d.org/) | Python/C++ | RANSAC plane fitting (`segment_plane`), DBSCAN clustering, voxel downsampling, normals, outlier removal, raycasting | `pip install open3d`. Best general entry point. |
| [PCL](https://pointclouds.org/) | C++ | Region growing (normal/RGB), Euclidean clustering, supervoxels (VCCS), LCCP, min-cut, SAC models (planes, cylinders, spheres) | Most complete classical toolbox. Heavy to build. Python bindings are patchy. |
| [CGAL](https://www.cgal.org/) | C++ (+ partial Python) | Shape detection (Efficient RANSAC, region growing), **Surface_mesh_segmentation (SDF-based part segmentation)**, Point Set Classification (random forest / ETHZ features) | Robust geometry. Good for **mesh part segmentation without deep learning**. GPL/LGPL. |
| [PyMeshLab](https://pymeshlab.readthedocs.io/) / [trimesh](https://trimesh.org/) / [libigl](https://libigl.github.io/) | Python | Connected components, curvature, geodesics, mesh cleaning, sampling meshes → point clouds | Glue code between meshes and point clouds. |
| [PDAL](https://pdal.io/) | C++/CLI/Python | SMRF / PMF ground filtering, cluster, outlier, `filters.classification` pipelines | Standard tooling for aerial and terrestrial LiDAR. |
| [CloudCompare](https://www.cloudcompare.org/) | GUI/CLI | Connected components, CANUPO / 3DMASC classifiers, RANSAC shape detection plugin, CSF ground filter | Useful for inspecting data and labeling by hand. |
| [pyransac3d](https://github.com/leomariga/pyRANSAC-3D) | Python | Plane, cylinder, sphere, and cuboid fitting | Lightweight. |

---

## 2. Learned semantic segmentation of point clouds (scenes)

### Pointcept ecosystem — **current reference codebase**
- **Repo:** [github.com/Pointcept/Pointcept](https://github.com/Pointcept/Pointcept)
- **Models included:** PTv1/v2/**v3**, **Sonata**, **Concerto**, OA-CNNs, Swin3D, MinkUNet, SpUNet, OctFormer, Stratified Transformer, plus PointGroup and other instance heads.
- **Datasets supported out of the box:** ScanNet, ScanNet200, ScanNet++, S3DIS, Matterport3D, Structured3D, SemanticKITTI, nuScenes, Waymo, ModelNet40.
- **Point Transformer V3 (CVPR'24 Oral).** It serializes the point cloud along space-filling curves instead of running a KNN neighbor search. That makes it faster and lighter on memory than PTv2, and the paper reports state of the art on more than 20 downstream tasks. A PTv3-based variant won the 2024 Waymo semantic segmentation challenge. ([paper](https://arxiv.org/abs/2312.10035), [repo](https://github.com/Pointcept/PointTransformerV3))
- **Sonata (CVPR'25 Highlight).** A self-supervised pre-trained PTv3 with a lightweight standalone package and **pretrained weights on Hugging Face**. It is the easiest way to get strong per-point features: run a linear probe or fine-tune with few labels. ([repo](https://github.com/facebookresearch/sonata), [HF](https://huggingface.co/collections/torch-pointcloud/sonata))
- **Concerto (NeurIPS'25).** Joint 2D–3D self-supervised learning, the successor to Sonata in the same family. The newer **Utonia** ("one encoder for all point clouds") is listed in the Pointcept org. Check its release status. ([Pointcept org](https://github.com/Pointcept))
- **Requirements:** NVIDIA GPU, CUDA, `spconv`, `torch-scatter`. `flash-attn` is optional but speeds things up.

### Other learned-segmentation frameworks
| Framework | Highlights | Status |
|---|---|---|
| [Open3D-ML](https://github.com/isl-org/Open3D-ML) | RandLA-Net, KPConv, PointTransformer, with pretrained models on SemanticKITTI, S3DIS, Toronto3D | Simple API, older models |
| [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) | PointNet++, PAConv, DGCNN, MinkUNet, Cylinder3D, SPVCNN | Maintenance has slowed |
| [OpenPCSeg](https://github.com/PJLab-ADG/PCSeg) | Outdoor LiDAR segmentation (MinkUNet, SPVCNN, Cylinder3D, RangeNet++) | Driving-focused |
| [Myria3D](https://github.com/IGNF/myria3d) | Aerial LiDAR classification (IGN France), ready-to-use pipeline | Built for aerial survey data |
| [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) | Strong classic baseline | Stable |
| `torch-points3d` | — | **Archived**, avoid |

---

## 3. 3D instance segmentation

| Method | Idea | Link |
|---|---|---|
| **OneFormer3D** (CVPR'24) | One transformer for semantic, instance, and panoptic segmentation | [github.com/filaPro/oneformer3d](https://github.com/filaPro/oneformer3d) |
| **Mask3D** (ICRA'23) | Transformer mask queries on sparse convolutions. Many open-vocabulary pipelines use it as the proposal generator | [github.com/JonasSchult/Mask3D](https://github.com/JonasSchult/Mask3D) |
| SPFormer / ISBNet | Superpoint / instance-aware kernels | GitHub |
| Pointcept (PTv3 + PointGroup) | Inside the same codebase as section 2 | Pointcept |

---

## 4. Zero-shot, promptable, and open-vocabulary point cloud segmentation

These methods need **no labeled training data**. They lift 2D foundation models (SAM, SAM 2, CLIP) into 3D.

| Method | Input | What it does | Link |
|---|---|---|---|
| **SAM2Point** | Point cloud (objects, indoor, outdoor) | Treats 3D data as multi-directional videos and runs SAM 2 with point, box, or mask prompts. Zero-shot, with no 2D–3D projection | [github.com/ZiyuGuo99/SAM2Point](https://github.com/ZiyuGuo99/SAM2Point), [paper](https://arxiv.org/abs/2408.16768) |
| **Point-SAM** | Point cloud | Native 3D promptable model distilled from SAM through a data engine. Works for parts and objects | [github.com/zyc00/Point-SAM](https://github.com/zyc00/Point-SAM), [paper](https://arxiv.org/abs/2406.17741) |
| **OpenMask3D** (NeurIPS'23) | Point cloud + posed RGB-D | Mask3D proposals plus CLIP features fused across views → text-queryable instances | [github.com/OpenMask3D/openmask3d](https://github.com/OpenMask3D/openmask3d) |
| **Open3DIS** (CVPR'24) | Point cloud + posed RGB-D | Adds 2D-mask-guided proposals. The authors report about 1.5× OpenMask3D's AP on ScanNet200 | [paper](https://arxiv.org/abs/2312.10671) |
| OpenIns3D, SAI3D, Segment3D | Variants | Variants of the same approach (some need no images) | GitHub |

**Caveat:** most open-vocabulary scene methods need **posed RGB(-D) images** alongside the point cloud. SAM2Point and Point-SAM work on points alone.

---

## 5. Mesh & object part segmentation

This area has moved quickly since 2024. Large models trained on Objaverse now segment *any* mesh into parts with no category labels.

| Method | Type | Notes | Link |
|---|---|---|---|
| **PartField** (NVIDIA, ICCV'25) | Feed-forward triplane feature field → clustering at multiple scales | Fast: seconds per shape, no per-shape optimization. Gives hierarchical parts and works on meshes and point clouds. The paper reports up to 20% higher accuracy than prior class-agnostic methods and is often orders of magnitude faster | [project](https://research.nvidia.com/labs/toronto-ai/partfield-release), [paper](https://arxiv.org/abs/2504.11451) |
| **P3-SAM** (Tencent Hunyuan3D-Part, 2025) | Native 3D promptable/automatic part segmentation | Trained on about 3.7M meshes with part masks. Code and weights are public. Takes any mesh as input | [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/P3-SAM), [HF](https://huggingface.co/tencent/Hunyuan3D-Part), [paper](https://arxiv.org/abs/2509.06784) |
| **PartSAM** (2025) | Promptable, trained natively on 3D | Reports 69.5 mIoU on PartObjaverse-Tiny, against SAMesh 56.9, SAMPart3D 53.5, and PartField (K-means) 51.5. These are self-reported numbers. **Public code not confirmed** | [paper](https://arxiv.org/abs/2509.21965) |
| **SAMPart3D** | SAM masks lifted to 3D, with optimization per shape | Slower, but the code and checkpoint are public | GitHub |
| **SAMesh** | Multi-view SAM + mesh connectivity | Tends to over-segment | GitHub |
| PartSLIP / PartSLIP++ | GLIP/SAM + few-shot | Semantic part labels from text prompts | GitHub |
| **DiffusionNet** | Learned operator on meshes and point clouds | Train your own supervised part segmentation. Robust to discretization | [github.com/nmwsharp/diffusion-net](https://github.com/nmwsharp/diffusion-net) |
| CGAL SDF segmentation | Classical | Splits shapes by shape-diameter function, needs no GPU | CGAL docs |

---

## 6. Datasets & benchmarks to evaluate against

- **Indoor scenes:** ScanNet / ScanNet200, **ScanNet++**, S3DIS, Matterport3D, Replica
- **Outdoor / LiDAR:** SemanticKITTI, nuScenes-lidarseg, Waymo, Toronto3D, DALES (aerial), SensatUrban
- **Objects / parts:** ShapeNetPart, PartNet, **PartObjaverse-Tiny**, PartNet-E (PartSLIP)
- **Metrics:** mIoU, overall accuracy (semantic); AP / AP50 / AP25 (instance); class-agnostic part mIoU (parts)

---

## 7. Practical constraints

- **GPU:** almost every learned method needs an NVIDIA GPU with CUDA. Expect at least 12–24 GB VRAM for inference on large scenes; Sonata/PTv3 scales with point count. CPU-only means classical methods only, or heavy downsampling.
- **Install friction:** `spconv`, `MinkowskiEngine`, `torch-scatter`, `flash-attn`, and custom CUDA ops are the usual pain points. Pin the exact PyTorch/CUDA versions each repo asks for, ideally one **conda env or Docker image per method**.
- **Licenses vary.** Many research repos are non-commercial (for example NVIDIA's research license or CC-BY-NC weights). CGAL is GPL. Check each one before any commercial use.
