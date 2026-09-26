#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec
scratch=/home/bjw-0/mosaic-rec-work/p2
python=/home/bjw-0/mosaic-rec-work/.venv/bin/python
monitor="$scratch/hpo/monitoring/lightgcn_t3_attempt2.log"
receipt="$repo/evidence/p2/p2-e00-lightgcn_t3.json"
trial_log="$scratch/hpo/logs/lightgcn_t3_attempt2.log"

if [[ -e "$monitor" ]]; then
  printf 'Preserve existing monitor artifact: %s\n' "$monitor" >&2
  exit 1
fi
mkdir -p "$(dirname "$monitor")"
kernel_line=$(dmesg | wc -l)

cd "$repo"
export PYTHONPATH=src
"$python" scripts/p2/run_hpo_queue.py \
  --repo . \
  --scratch "$scratch" \
  --lock "$scratch/locks/heavy_gpu.lock" &
queue_pid=$!

while kill -0 "$queue_pid" 2>/dev/null && [[ ! -e "$receipt" ]]; do
  {
    printf 'TIMESTAMP %s queue_pid=%s\n' "$(date -u +%FT%TZ)" "$queue_pid"
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader
    ps -o pid,etimes,stat,cmd -p "$queue_pid"
    pgrep -af 'LIGHTGCN_T3|run_hpo_queue.py' || true
    current_line=$(dmesg | wc -l)
    if (( current_line > kernel_line )); then
      dmesg --time-format iso | sed -n "$((kernel_line + 1)),${current_line}p"
      kernel_line=$current_line
    fi
    if [[ -e "$trial_log" ]]; then
      grep -Ein 'cuda|device.not.ready|error|traceback' "$trial_log" || true
    fi
  } >>"$monitor" 2>&1
  sleep 10
done

set +e
wait "$queue_pid"
queue_status=$?
set -e
{
  printf 'MONITOR_END %s queue_status=%s receipt_exists=%s\n' \
    "$(date -u +%FT%TZ)" "$queue_status" "$([[ -e "$receipt" ]] && echo true || echo false)"
  current_line=$(dmesg | wc -l)
  if (( current_line > kernel_line )); then
    dmesg --time-format iso | sed -n "$((kernel_line + 1)),${current_line}p"
  fi
  if [[ -e "$trial_log" ]]; then
    grep -Ein 'cuda|device.not.ready|error|traceback' "$trial_log" || true
  fi
} >>"$monitor" 2>&1
exit "$queue_status"
