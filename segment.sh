#!/usr/bin/env bash
# Segment point clouds and meshes. Usage:
#   ./segment.sh my_scan.las            one file
#   ./segment.sh my_folder/             every .ply .las .laz .txt .obj .glb in a folder
# Results go to results/<name>/  (open the .ply files in CloudCompare or MeshLab)
# Extra options are passed through, e.g.  ./segment.sh facade.obj --up y
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
CONDA=$(command -v mamba || command -v conda)
IN="$1"; shift

run_one() {
  name=$(basename "${1%.*}")
  echo "=== $name ==="
  $CONDA run -n heritage-sam3 --no-capture-output \
    python "$REPO/scripts/zero_shot_segment.py" "$1" "$REPO/results/$name" --views both --save-views "${@:2}"
}

if [ -d "$IN" ]; then
  find "$IN" -maxdepth 1 -type f \( -iname '*.ply' -o -iname '*.las' -o -iname '*.laz' \
       -o -iname '*.txt' -o -iname '*.obj' -o -iname '*.glb' \) -print0 | sort -z |
  while IFS= read -r -d '' f; do run_one "$f" "$@"; done
else
  run_one "$IN" "$@"
fi
echo "Done. Open results/<name>/points_labelled.ply (and mesh_labelled.ply for meshes)."
