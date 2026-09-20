#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
SCRATCH=/home/bjw-0/mosaic-rec-work
PY="$SCRATCH/.venv/bin/python"
cd "$ROOT"
mkdir -p "$SCRATCH/checkpoints" "$SCRATCH/logs"

for architecture in sasrec gru4rec; do
  receipt="$ROOT/evidence/microlens_${architecture}_coldguard_p1.json"
  checkpoint="$SCRATCH/checkpoints/microlens_${architecture}_coldguard_p1.pt"
  log="$SCRATCH/logs/microlens_${architecture}_coldguard_p1.log"
  if [[ -e "$receipt" || -e "$checkpoint" ]]; then
    echo "HOLD: existing $architecture coldguard output requires inspection" >&2
    exit 4
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "HOLD: another GPU process is active before $architecture" >&2
    exit 3
  fi
  "$PY" -m mosaic_rec.pilot_sasrec \
    "$SCRATCH/data/microlens-100k/split_p1" \
    "$SCRATCH/data/microlens-100k/pairs.csv" \
    "$receipt" --architecture "$architecture" \
    --lock "$SCRATCH/locks/gpu.lock" --checkpoint "$checkpoint" \
    --manifest "$ROOT/configs/sequence_coldguard_pilot_p1.json" \
    --epochs 3 --seed 20260920 > "$log" 2>&1
  echo "PASS $architecture coldguard pilot receipt: $receipt"
done
