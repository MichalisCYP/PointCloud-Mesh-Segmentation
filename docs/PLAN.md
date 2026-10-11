# Plan

1. **Install once:** `bash install.sh` (after getting SAM 3 access, see README).
2. **Run:** `./segment.sh <file or folder>`.
3. **Open** `results/<name>/*.ply` in CloudCompare or MeshLab.

That's it. The method renders the building from many viewpoints, asks SAM 3 to find each class by name, and transfers the answers back to the 3D points and mesh faces.

## If something looks off
| Problem | Add this to the command |
|---|---|
| Mesh or scan sideways / upside down | `--up y` |
| Missing elements | `--threshold 0.3` |
| Too many wrong detections | `--threshold 0.5` |
| Out of GPU memory | `--size 768` |
| Want different classes | `--classes my_classes.json` (example below) |

```json
[
  {"name": "column", "prompts": ["column", "pillar"], "weight": 1.0},
  {"name": "window", "prompts": ["window"],           "weight": 1.0},
  {"name": "wall",   "prompts": ["wall"],             "weight": 0.6},
  {"name": "other",  "prompts": [],                   "weight": 1.0}
]
```

Optional, only if you later want higher accuracy by training: [TRAINING.md](TRAINING.md). The survey of all methods is in [RESEARCH.md](RESEARCH.md).
