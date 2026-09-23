#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
scratch=/home/bjw-0/mosaic-rec-work/p2
router=${1:?usage: run_gate_queue.sh R3|R4 attempt}
attempt=${2:?usage: run_gate_queue.sh R3|R4 attempt}
case "$router" in
  R3) stem=r3_context_aware_pilot ; stage=d03 ;;
  R4) stem=r4_warm_preserving_pilot ; stage=d04 ;;
  *) echo "router must be R3 or R4" >&2; exit 2 ;;
esac
checkpoint="$scratch/checkpoints/${stem}_${attempt}.pt"
output="$scratch/development/${stem}_${attempt}_validation.parquet"
receipt="$repo/evidence/p2/p2-${stage}-${router,,}-${attempt}.json"
log="$scratch/logs/${stem}_${attempt}.log"
for path in "$checkpoint" "$output" "$receipt" "$log"; do
  [[ ! -e "$path" ]] || { echo "REFUSE_EXISTING=$path" >&2; exit 17; }
done
mkdir -p "$scratch/logs"
idle=0
while [[ "$idle" -lt 6 ]]; do
  activity=$(nvidia-smi \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ')
  memory=${activity%,*}
  utilization=${activity#*,}
  # CPU-only preprocessing may overlap p2. The inner Python lock repeats a
  # compute-PID, memory, and utilization qualification before CUDA model load.
  if [[ "$memory" -le 256 ]] \
    && [[ "$utilization" -le 5 ]]; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  printf 'WAIT_GPU %s memory_mib=%s utilization=%s idle_checks=%s/6\n' \
    "$(date -u +%FT%TZ)" "$memory" "$utilization" "$idle"
  sleep 5
done
cd "$repo"
export PYTHONPATH=.:src
/home/bjw-0/mosaic-rec-work/.venv/bin/python scripts/p2/run_gate_pilot.py "$router" \
  --repo . \
  --ladder configs/p2/router_ladder.json \
  --pilot-config configs/p2/development_pilots_attempt2.json \
  --fit "$scratch/data/pixelrec50k/FIT_VISIBLE.parquet" \
  --queries "$scratch/contexts/pixelrec50k/VALIDATION_QUERIES.parquet" \
  --contexts "$scratch/contexts/pixelrec50k" \
  --features "$scratch/features/pixelrec50k" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --receipt "$receipt" \
  --lock "$scratch/locks/heavy_gpu.lock" 2>&1 | tee "$log"
