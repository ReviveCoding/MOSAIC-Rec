#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
scratch=/home/bjw-0/mosaic-rec-work/p2
idle=0
while [[ "$idle" -lt 6 ]]; do
  if ! activity=$(nvidia-smi \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '); then
    idle=0
    printf 'WAIT_GPU %s probe_failed=gpu_activity idle_checks=0/6\n' \
      "$(date -u +%FT%TZ)"
    sleep 5
    continue
  fi
  memory=${activity%,*}
  utilization=${activity#*,}
  if ! compute_pids=$(nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'); then
    idle=0
    printf 'WAIT_GPU %s probe_failed=compute_pids idle_checks=0/6\n' \
      "$(date -u +%FT%TZ)"
    sleep 5
    continue
  fi
  if [[ -z "$compute_pids" ]] \
    && [[ "$memory" -le 256 ]] \
    && [[ "$utilization" -le 5 ]]; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  printf 'WAIT_GPU %s memory_mib=%s utilization=%s compute_pids=%s idle_checks=%s/6\n' \
    "$(date -u +%FT%TZ)" "$memory" "$utilization" \
    "${compute_pids:-none}" "$idle"
  sleep 5
done

cd "$repo"
export PYTHONPATH=src
exec /home/bjw-0/mosaic-rec-work/.venv/bin/python \
  scripts/p2/run_alignment_queue.py \
  --repo . \
  --scratch "$scratch" \
  --lock "$scratch/locks/heavy_gpu.lock"
