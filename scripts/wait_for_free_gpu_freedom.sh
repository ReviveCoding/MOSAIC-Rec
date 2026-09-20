#!/usr/bin/env bash
set -euo pipefail

# The sole GPU is shared with unrelated local projects. Require five minutes
# of continuous absence of compute processes before the distinct retry.
required_checks=11
free_checks=0
while (( free_checks < required_checks )); do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' || \
    pgrep -af 'act_agent\.(training|evaluation)|contextweave\.model_pilot' >/dev/null; then
    free_checks=0
  else
    free_checks=$((free_checks + 1))
  fi
  if (( free_checks < required_checks )); then
    sleep 30
  fi
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' || \
  pgrep -af 'act_agent\.(training|evaluation)|contextweave\.model_pilot' >/dev/null; then
  echo 'HOLD: GPU was reclaimed during the final check' >&2
  exit 3
fi

exec bash /mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec/scripts/run_freedom_reference_attempt2.sh
