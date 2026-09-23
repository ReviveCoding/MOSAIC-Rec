"""Run remaining p2 development baseline pilots serially and crash safely."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file

BASELINES = (
    ("BPR_MF", "bpr_mf_pilot", "p2-c00-bpr-mf-pilot.json"),
    ("LIGHTGCN", "lightgcn_pilot", "p2-c01-lightgcn-pilot.json"),
    ("SASREC", "sasrec_pilot", "p2-c03-sasrec-pilot.json"),
    ("VBPR", "vbpr_pilot", "p2-c04-vbpr-pilot.json"),
)


def verify_terminal(receipt_path: Path, checkpoint: Path, output: Path) -> None:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PASS":
        raise RuntimeError(f"Non-PASS development receipt: {receipt_path}")
    if not checkpoint.is_file() or sha256_file(checkpoint) != receipt.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError(f"Development checkpoint hash mismatch: {checkpoint}")
    if not output.is_file() or sha256_file(output) != receipt["validation"].get(
        "output_sha256"
    ):
        raise RuntimeError(f"Development output hash mismatch: {output}")


def run_one(
    *,
    command: list[str],
    log: Path,
    receipt: Path,
    checkpoint: Path,
    output: Path,
    environment: dict[str, str],
) -> None:
    if receipt.exists():
        verify_terminal(receipt, checkpoint, output)
        print(f"SKIP_TERMINAL {receipt.name}", flush=True)
        return
    partials = [str(path) for path in (log, checkpoint, output) if path.exists()]
    if partials:
        raise RuntimeError(
            f"Audit partial development artifacts before retry: {partials}"
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as handle:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Development job exited {completed.returncode}; audit {log}"
        )
    verify_terminal(receipt, checkpoint, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    scratch = args.scratch.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    common = [
        "--repo",
        str(repo),
        "--fit",
        str(scratch / "data/pixelrec50k/FIT_VISIBLE.parquet"),
        "--queries",
        str(scratch / "contexts/pixelrec50k/VALIDATION_QUERIES.parquet"),
        "--contexts",
        str(scratch / "contexts/pixelrec50k"),
        "--features",
        str(scratch / "features/pixelrec50k"),
        "--lock",
        str(args.lock),
    ]
    for baseline, stem, receipt_name in BASELINES:
        checkpoint = scratch / f"checkpoints/{stem}.pt"
        output = scratch / f"development/{stem}_validation.parquet"
        receipt = repo / f"evidence/p2/{receipt_name}"
        command = [
            sys.executable,
            str(repo / "scripts/p2/run_additional_baseline.py"),
            baseline,
            "--config",
            str(repo / "configs/p2/additional_baseline_pilots.json"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--receipt",
            str(receipt),
            *common,
        ]
        print(f"START {baseline}", flush=True)
        run_one(
            command=command,
            log=scratch / f"logs/{stem}.log",
            receipt=receipt,
            checkpoint=checkpoint,
            output=output,
            environment=environment,
        )
        print(f"PASS {baseline}", flush=True)
    stem = "legacy_m1_pilot"
    checkpoint = scratch / f"checkpoints/{stem}.pt"
    output = scratch / f"development/{stem}_validation.parquet"
    receipt = repo / "evidence/p2/p2-d00-legacy-m1-pilot.json"
    command = [
        sys.executable,
        str(repo / "scripts/p2/run_legacy_pilot.py"),
        "--config",
        str(repo / "configs/p2/legacy_m1_pilot.json"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--receipt",
        str(receipt),
        *common,
    ]
    print("START LEGACY_M1", flush=True)
    run_one(
        command=command,
        log=scratch / f"logs/{stem}.log",
        receipt=receipt,
        checkpoint=checkpoint,
        output=output,
        environment=environment,
    )
    print("PASS LEGACY_M1", flush=True)


if __name__ == "__main__":
    main()
