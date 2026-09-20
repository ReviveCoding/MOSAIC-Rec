#!/usr/bin/env bash
set -euo pipefail

PY=/home/bjw-0/mosaic-rec-work/.venv/bin/python
ROOT=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
SCRATCH=/home/bjw-0/mosaic-rec-work

for variant in 1 2 3 4 5 6 7; do
  echo "Starting M${variant} on CUDA"
  "$PY" -m mosaic_rec.pilot_mosaic "$variant" \
    "$SCRATCH/data/microlens-100k/split_p1" \
    "$SCRATCH/data/microlens-100k/pairs.csv" \
    "$SCRATCH/features/microlens-100k" \
    "$SCRATCH/data/microlens-100k/tags_to_summary.csv" \
    "$SCRATCH/data/microlens-100k/title_en.csv" \
    "$ROOT/evidence/microlens_m${variant}_pilot.json" \
    --lock "$SCRATCH/locks/gpu.lock" \
    --checkpoint "$SCRATCH/checkpoints/microlens_m${variant}_pilot.pt" \
    --epochs 3
done
