# Heritage Point Cloud & Mesh Segmentation

Labels every point/face of a heritage building as **arch, column, moldings, floor, door/window, wall, stairs, vault, roof, other**.
No training, no labelled data. Needs a PC with an NVIDIA GPU (8 GB is enough) running Linux or WSL2.

## 1. Install (once)
1. Ask for access to the SAM 3 model: <https://huggingface.co/facebook/sam3>. Once accepted, run `hf auth login`.
2. Then:
   ```bash
   git clone https://github.com/michaliscyp/pointcloud-mesh-segmentation.git
   cd pointcloud-mesh-segmentation
   bash install.sh
   ```

## 2. Run
```bash
./segment.sh my_scan.las        # one file  (.ply .las .laz .txt .obj .glb)
./segment.sh my_folder/         # every file in a folder
```

## 3. Results
`results/<name>/points_labelled.ply` (and `mesh_labelled.ply` for meshes). Open them in CloudCompare or MeshLab; colours = classes, the `label` field holds the class id.

**Mesh lying on its side?** Run `./segment.sh file.obj --up y`.

More details: [docs/PLAN.md](docs/PLAN.md)
