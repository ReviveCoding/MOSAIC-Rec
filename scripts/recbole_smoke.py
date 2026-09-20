"""Reference runtime smoke on FIT-only interactions; not scientific parity."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from recbole.quick_start import run_recbole


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", choices=("SASRec", "BERT4Rec"), default="SASRec")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for RecBole neural reference smoke")
    # RecBole 1.2.1 does not pass weights_only to torch.load. This process
    # creates and reloads only its own checkpoint in a private scratch path.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    with args.lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        lock_handle.write(f"pid={__import__('os').getpid()} model=RECBOLE_{args.model.upper()}_SMOKE\n")
        lock_handle.flush()
        result = run_recbole(model=args.model, dataset="microlens_p1",
                             config_file_list=[str(args.config)], saved=True)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    receipt = {
        "stage_id": "C05", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT", "status": "REFERENCE_RUNTIME_SMOKE_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "reference": f"RecBole 1.2.1 {args.model}", "device": torch.cuda.get_device_name(0),
        "result": result,
        "warnings": ["RecBole internally splits FIT-only interactions for smoke; these results are not comparable to the frozen protocol and cannot select a comparator."],
        "evidence_class": "reference_runtime_smoke",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
