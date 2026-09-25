#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
scratch=/home/bjw-0/mosaic-rec-work/p2
export PYTHONPATH="$repo/src"
python=/home/bjw-0/mosaic-rec-work/.venv/bin/python
until "$python" -c 'from mosaic_rec.p2.gpu import sustained_idle; sustained_idle()'; do
  printf 'WAIT_GPU %s sustained_idle_probe_failed\n' "$(date -u +%FT%TZ)"
  sleep 5
done

cd "$repo"
exec "$python" \
  scripts/p2/run_alignment_queue.py \
  --repo . \
  --scratch "$scratch" \
  --lock "$scratch/locks/heavy_gpu.lock"
