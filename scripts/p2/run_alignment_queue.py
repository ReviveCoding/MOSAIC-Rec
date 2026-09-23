"""Run preregistered p2 score-alignment trials serially and crash safely."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    scratch = args.scratch.resolve()
    manifest = json.loads(
        (repo / "configs/p2/alignment_trials_manifest.json").read_text()
    )
    environment = os.environ.copy()
    source = str(repo / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    for identity in manifest["trials"]:
        alignment_id = identity["alignment_id"]
        config_path = repo / identity["path"]
        if sha256_file(config_path) != identity["sha256"]:
            raise RuntimeError(f"Alignment config hash mismatch: {alignment_id}")
        stem = alignment_id.lower()
        output = scratch / f"alignment/{stem}_validation.parquet"
        receipt = repo / f"evidence/p2/p2-b03-{stem}.json"
        log = scratch / f"alignment/logs/{stem}.log"
        if receipt.exists():
            payload = json.loads(receipt.read_text())
            if payload.get("status") != "PASS":
                raise RuntimeError(f"Non-PASS alignment receipt: {receipt}")
            if not output.is_file() or sha256_file(output) != payload["validation"].get(
                "output_sha256"
            ):
                raise RuntimeError(f"Alignment output hash mismatch: {alignment_id}")
            print(f"SKIP_TERMINAL {alignment_id}", flush=True)
            continue
        partials = [str(path) for path in (output, log) if path.exists()]
        if partials:
            raise RuntimeError(
                f"Audit partial alignment artifacts before retry: {partials}"
            )
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(repo / "scripts/p2/run_alignment_trial.py"),
            alignment_id,
            "--repo",
            str(repo),
            "--scratch",
            str(scratch),
            "--lock",
            str(args.lock),
        ]
        print(f"START {alignment_id}", flush=True)
        with log.open("xb") as handle:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Alignment {alignment_id} exited {completed.returncode}; audit {log}"
            )
        payload = json.loads(receipt.read_text())
        if payload.get("status") != "PASS" or sha256_file(output) != payload[
            "validation"
        ].get("output_sha256"):
            raise RuntimeError(f"Terminal validation failed: {alignment_id}")
        print(f"PASS {alignment_id}", flush=True)


if __name__ == "__main__":
    main()
