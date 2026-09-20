#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
SCRATCH=/home/bjw-0/mosaic-rec-work
PY="$SCRATCH/.venv/bin/python"
RECEIPT="$ROOT/evidence/microlens_bert4rec_common_pilot_p1.json"
CHECKPOINT="$SCRATCH/checkpoints/microlens_bert4rec_common_pilot_p1.pt"
LOG="$SCRATCH/logs/microlens_bert4rec_common_pilot_p1.log"
cd "$ROOT"
mkdir -p "$SCRATCH/checkpoints" "$SCRATCH/logs"

if [[ -e "$RECEIPT" || -e "$CHECKPOINT" ]]; then
  echo 'HOLD: BERT4Rec pilot output already exists' >&2
  exit 4
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'HOLD: another GPU process is active' >&2
  exit 3
fi

"$PY" -m mosaic_rec.pilot_bert4rec \
  "$SCRATCH/data/microlens-100k/split_p1" \
  "$SCRATCH/data/microlens-100k/pairs.csv" \
  "$RECEIPT" --checkpoint "$CHECKPOINT" \
  --lock "$SCRATCH/locks/gpu.lock" \
  --config "$ROOT/configs/bert4rec_common_pilot_p1.json" \
  --epochs 3 --seed 20260920 > "$LOG" 2>&1
echo "BERT4Rec pilot receipt: $RECEIPT"
