#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
scratch=/home/bjw-0/mosaic-rec-work/p2
model=${1:?usage: run_corrected_expert_queue.sh sequence|content}
if [[ "$model" != sequence && "$model" != content ]]; then
  echo "model must be sequence or content" >&2
  exit 2
fi

if [[ "$model" == sequence ]]; then
  stem=cseq_bert4rec_pilot_attempt2
else
  stem=ccold_two_tower_pilot_attempt2
fi
checkpoint="$scratch/checkpoints/$stem.pt"
output="$scratch/development/${stem}_validation.parquet"
receipt="$repo/evidence/p2/p2-${stem//_/-}.json"
log="$scratch/logs/$stem.log"

for path in "$checkpoint" "$output" "$receipt" "$log"; do
  if [[ -e "$path" ]]; then
    echo "REFUSE_EXISTING=$path" >&2
    exit 17
  fi
done
mkdir -p "$scratch/logs"

idle=0
while [[ "$idle" -lt 6 ]]; do
  activity=$(nvidia-smi \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ')
  memory=${activity%,*}
  utilization=${activity#*,}
  if ! pgrep -f 'act_agent\.evaluation|ContextWeave' >/dev/null \
    && [[ "$memory" -le 256 ]] \
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
/home/bjw-0/mosaic-rec-work/.venv/bin/python \
  scripts/p2/run_expert_pilot.py "$model" \
  --repo . \
  --config configs/p2/development_pilots_attempt2.json \
  --fit "$scratch/data/pixelrec50k/FIT_VISIBLE.parquet" \
  --queries "$scratch/contexts/pixelrec50k/VALIDATION_QUERIES.parquet" \
  --contexts "$scratch/contexts/pixelrec50k" \
  --features "$scratch/features/pixelrec50k" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --receipt "$receipt" \
  --lock "$scratch/locks/heavy_gpu.lock" 2>&1 | tee "$log"
