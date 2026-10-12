"""Convert a mesh Python/trimesh can't read (e.g. FBX) to GLB using Blender.

Runs headless inside Blender; called automatically by zero_shot_segment.py:
    blender -b --factory-startup --python blender_convert.py -- input.fbx output.glb

The GLB is written Z-up (Blender's convention) with textures embedded.
"""
import sys

import bpy

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
ext = src.lower().rsplit(".", 1)[-1]
if ext == "fbx":
    bpy.ops.import_scene.fbx(filepath=src)
elif ext in ("dae",):
    bpy.ops.wm.collada_import(filepath=src)
elif ext in ("usd", "usda", "usdc", "usdz"):
    bpy.ops.wm.usd_import(filepath=src)
elif ext == "abc":
    bpy.ops.wm.alembic_import(filepath=src)
else:
    raise SystemExit(f"blender_convert: unsupported format .{ext}")
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_yup=False, export_apply=True)
print(f"blender_convert: {src} -> {dst}")
