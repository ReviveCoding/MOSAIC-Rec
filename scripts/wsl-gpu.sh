#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/bjw-0/mosaic-rec-work"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME="${HF_HOME:-$WORK_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK_ROOT/cache/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK_ROOT/cache/uv}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"

# Avoid uncontrolled nested tokenizer multiprocessing.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$UV_CACHE_DIR" \
  "$TMPDIR" \
  "$WORK_ROOT/logs" \
  "$WORK_ROOT/locks"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi not available in WSL." >&2
  exit 91
}

nvidia-smi -L >/dev/null

if [[ $# -eq 0 ]]; then
    exec bash
fi

exec "$@"