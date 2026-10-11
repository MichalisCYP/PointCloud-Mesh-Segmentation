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

if [ -d "$IN" ]; then
  FILES=$(find "$IN" -maxdepth 1 -type f \( -iname '*.ply' -o -iname '*.las' -o -iname '*.laz' \
          -o -iname '*.txt' -o -iname '*.obj' -o -iname '*.glb' \) | sort)
else
  FILES="$IN"
fi

for f in $FILES; do
  name=$(basename "${f%.*}")
  echo "=== $name ==="
  $CONDA run -n heritage-sam3 --no-capture-output \
    python "$REPO/scripts/zero_shot_segment.py" "$f" "$REPO/results/$name" --views both --save-views "$@"
done
echo "Done. Open results/<name>/points_labelled.ply (and mesh_labelled.ply for meshes)."
