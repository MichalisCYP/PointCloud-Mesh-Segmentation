# Heritage Point Cloud & Mesh Segmentation

Labels every point/face of a heritage building as **arch, column, moldings, floor, door/window, wall, stairs, vault, roof, other**.
No training, no labelled data. Needs a PC with an NVIDIA GPU (8 GB is enough).

Inputs: point clouds `.ply .las .laz .txt`, meshes `.obj .glb .gltf .ply`, and `.fbx` (converted automatically with Blender, which must be installed).

## 1. Install (once)
1. Ask for access to the SAM 3 model: <https://huggingface.co/facebook/sam3> (needs a free Hugging Face account; approval can take a few hours).
2. Install:
   - **Windows:** double-click `install.bat` (or run it in a terminal).
   - **Linux / WSL2:** `bash install.sh`
3. Once access is approved, log in (paste a *read* token from huggingface.co → Settings → Access Tokens):
   - Windows: `.venv\Scripts\hf.exe auth login`
   - Linux: `hf auth login`

## 2. Run
```bat
segment.bat my_scan.las          :: Windows, one file
segment.bat D:\scans             :: every file in a folder
```
```bash
./segment.sh my_scan.las         # Linux / WSL2
```

## 3. Results
`results\<name>\points_labelled.ply` (and `mesh_labelled.ply` for meshes). Open them in CloudCompare or MeshLab; colours = classes, the `label` field holds the class id. `results\<name>\views\` shows what SAM 3 saw in each view.

**Mesh lying on its side?** Add `--up y` (not needed for FBX, which Blender already converts to Z-up).

More details: [docs/PLAN.md](docs/PLAN.md)
