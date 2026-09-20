#!/usr/bin/env bash
set -euo pipefail

PY=/home/bjw-0/mosaic-rec-work/.venv/bin/python
ROOT=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
SCRATCH=/home/bjw-0/mosaic-rec-work

for seed in 20260921 20260922; do
  for variant in 1 4 6 7; do
    receipt="$ROOT/evidence/microlens_m${variant}_seed${seed}_development.json"
    checkpoint="$SCRATCH/checkpoints/microlens_m${variant}_seed${seed}_development.pt"
    if [[ -e "$receipt" || -e "$checkpoint" ]]; then
      if [[ ! -s "$receipt" || ! -s "$checkpoint" ]]; then
        echo "HOLD: incomplete existing run M${variant} seed ${seed}; inspect before resuming" >&2
        exit 4
      fi
      "$PY" -c 'import hashlib,json,sys; r=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert h==r["checkpoint_sha256"]' "$receipt" "$checkpoint"
      echo "Verified existing M${variant} seed ${seed}; skipping"
      continue
    fi
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
      echo "HOLD: another GPU process is active before M${variant} seed ${seed}" >&2
      exit 3
    fi
    echo "Starting M${variant} seed ${seed} on CUDA"
    "$PY" -m mosaic_rec.pilot_mosaic "$variant" \
      "$SCRATCH/data/microlens-100k/split_p1" \
      "$SCRATCH/data/microlens-100k/pairs.csv" \
      "$SCRATCH/features/microlens-100k" \
      "$SCRATCH/data/microlens-100k/tags_to_summary.csv" \
      "$SCRATCH/data/microlens-100k/title_en.csv" \
      "$receipt" \
      --lock "$SCRATCH/locks/gpu.lock" \
      --checkpoint "$checkpoint" \
      --epochs 3 --seed "$seed" \
      > "$SCRATCH/logs/microlens_m${variant}_seed${seed}_development.log" 2>&1
  done
done
