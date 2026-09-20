#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
SCRATCH=/home/bjw-0/mosaic-rec-work
PY="$SCRATCH/.venv/bin/python"
FEATURES="$SCRATCH/references/mmrec_micro_p1_freedom_attempt2"
RECEIPT="$ROOT/evidence/microlens_freedom_mmrec_reference_attempt2.json"
CHECKPOINT="$SCRATCH/checkpoints/microlens_freedom_mmrec_reference_attempt2.pt"
LOG="$SCRATCH/logs/microlens_freedom_mmrec_reference_attempt2.log"
cd "$ROOT"

if [[ -e "$RECEIPT" || -e "$CHECKPOINT" || -e "$FEATURES/mm_adj_freedomdsp_10_1.pt" ]]; then
  echo "HOLD: attempt-2 output exists; inspect before rerun" >&2
  exit 4
fi
if [[ ! -s "$FEATURES/vision.npy" || ! -s "$FEATURES/text.npy" ]]; then
  echo "HOLD: attempt-2 input links are missing" >&2
  exit 5
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "HOLD: another GPU compute process is active" >&2
  exit 3
fi

"$PY" -m mosaic_rec.pilot_mmrec_freedom \
  "$SCRATCH/references/MMRec" "$FEATURES" \
  "$SCRATCH/data/microlens-100k/split_p1" \
  "$SCRATCH/data/microlens-100k/pairs.csv" "$RECEIPT" \
  --checkpoint "$CHECKPOINT" --lock "$SCRATCH/locks/gpu.lock" \
  --epochs 1 --seed 20260920 --attempt 2 > "$LOG" 2>&1
echo "Attempt-2 receipt: $RECEIPT"
