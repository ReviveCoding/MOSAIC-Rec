#!/usr/bin/env bash
set -u -o pipefail

ROOT=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
SCRATCH=/home/bjw-0/mosaic-rec-work
PY="$SCRATCH/.venv/bin/python"
cd "$ROOT" || exit 2
mkdir -p "$SCRATCH/evidence" "$SCRATCH/logs"

failed=0
for variant in 1 4 6 7; do
  receipt="$ROOT/evidence/microlens_m${variant}_validation_replay.json"
  output="$SCRATCH/evidence/microlens_m${variant}_validation_replay.parquet"
  log="$SCRATCH/logs/microlens_m${variant}_validation_replay.log"
  if [[ -e "$receipt" || -e "$output" ]]; then
    echo "HOLD M${variant}: existing replay artifact requires review" >&2
    failed=1
    continue
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "HOLD M${variant}: another GPU process is active" >&2
    exit 3
  fi
  "$PY" -m mosaic_rec.score_mosaic_validation "$variant" \
    "$ROOT/evidence/microlens_m${variant}_pilot.json" \
    "$SCRATCH/data/microlens-100k/split_p1" \
    "$SCRATCH/data/microlens-100k/pairs.csv" \
    "$SCRATCH/features/microlens-100k" \
    "$SCRATCH/data/microlens-100k/tags_to_summary.csv" \
    "$SCRATCH/data/microlens-100k/title_en.csv" \
    "$receipt" --output "$output" --lock "$SCRATCH/locks/gpu.lock" \
    > "$log" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "PASS M${variant} validation replay"
  else
    echo "FAIL M${variant} validation replay: exit $status; inspect $log" >&2
    failed=1
  fi
done
exit "$failed"
