#!/usr/bin/env bash
set -u -o pipefail

ROOT=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
SCRATCH=/home/bjw-0/mosaic-rec-work
PY="$SCRATCH/.venv/bin/python"
cd "$ROOT" || exit 2
mkdir -p "$SCRATCH/logs" "$SCRATCH/checkpoints"

run_one() {
  local family="$1" module="$2" receipt="$3" checkpoint="$4" log="$5"
  if [[ -e "$receipt" || -e "$checkpoint" ]]; then
    echo "HOLD $family: existing receipt or checkpoint requires review; no overwrite" >&2
    return 4
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "HOLD $family: another GPU compute process is active" >&2
    return 3
  fi
  echo "Starting $family CUDA reference runtime pilot"
  "$PY" -m "$module" \
    "$SCRATCH/references/MMRec" \
    "$SCRATCH/references/mmrec_micro_p1" \
    "$SCRATCH/data/microlens-100k/split_p1" \
    "$SCRATCH/data/microlens-100k/pairs.csv" \
    "$receipt" \
    --checkpoint "$checkpoint" \
    --lock "$SCRATCH/locks/gpu.lock" \
    --epochs 1 --seed 20260920 > "$log" 2>&1
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo "FAIL $family: exit $exit_code; inspect $log and $receipt" >&2
  else
    echo "PASS $family: $receipt"
  fi
  return "$exit_code"
}

run_one BM3 mosaic_rec.pilot_mmrec_bm3 \
  "$ROOT/evidence/microlens_bm3_mmrec_reference_pilot.json" \
  "$SCRATCH/checkpoints/microlens_bm3_mmrec_reference_pilot.pt" \
  "$SCRATCH/logs/microlens_bm3_mmrec_reference_pilot.log"
bm3_status=$?

run_one FREEDOM mosaic_rec.pilot_mmrec_freedom \
  "$ROOT/evidence/microlens_freedom_mmrec_reference_pilot.json" \
  "$SCRATCH/checkpoints/microlens_freedom_mmrec_reference_pilot.pt" \
  "$SCRATCH/logs/microlens_freedom_mmrec_reference_pilot.log"
freedom_status=$?

echo "BM3 exit=$bm3_status FREEDOM exit=$freedom_status"
if [[ $bm3_status -eq 0 && $freedom_status -eq 0 ]]; then
  exit 0
fi
exit 1
