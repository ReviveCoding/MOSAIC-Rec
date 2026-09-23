"""Run the frozen p2 E01 independent-seed queue serially and crash safely."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file

FAMILY_ORDER = (
    "CSEQ_BERT4REC",
    "CCOLD_TWO_TOWER",
    "LEGACY_M1",
    "BPR_MF",
    "LIGHTGCN",
    "SASREC",
    "VBPR",
    "R2",
    "R3",
    "R4",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    scratch = args.scratch.resolve()
    manifest = json.loads(args.manifest.read_text())
    runs = {run["run_id"]: run for run in manifest["runs"]}
    seeds = sorted({int(run["seed"]) for run in runs.values()})
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    for seed in seeds:
        for family in FAMILY_ORDER:
            matching = [
                run
                for run in runs.values()
                if int(run["seed"]) == seed and run["family"] == family
            ]
            if len(matching) != 1:
                raise RuntimeError(f"Expected one E01 run for {family} seed {seed}")
            identity = matching[0]
            run_id = identity["run_id"]
            stem = run_id.lower()
            config_path = Path(identity["path"])
            if not config_path.is_absolute():
                config_path = repo / config_path
            if sha256_file(config_path) != identity["sha256"]:
                raise RuntimeError(f"E01 config hash mismatch: {run_id}")
            checkpoint = scratch / f"seed_qualification/checkpoints/{stem}.pt"
            output = scratch / f"seed_qualification/results/{stem}_validation.parquet"
            receipt = repo / f"evidence/p2/p2-e01-{stem}.json"
            log = scratch / f"seed_qualification/logs/{stem}.log"
            if receipt.exists():
                payload = json.loads(receipt.read_text())
                if payload.get("status") != "PASS":
                    raise RuntimeError(f"Non-PASS E01 receipt: {receipt}")
                if sha256_file(checkpoint) != payload.get("checkpoint_sha256"):
                    raise RuntimeError(f"E01 checkpoint hash mismatch: {run_id}")
                if sha256_file(output) != payload["validation"].get("output_sha256"):
                    raise RuntimeError(f"E01 output hash mismatch: {run_id}")
                print(f"SKIP_TERMINAL {run_id}", flush=True)
                continue
            partials = [
                str(path) for path in (checkpoint, output, log) if path.exists()
            ]
            if partials:
                raise RuntimeError(f"Audit partial E01 artifacts: {partials}")
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(repo / "scripts/p2/run_seed_trial.py"),
                run_id,
                "--repo",
                str(repo),
                "--scratch",
                str(scratch),
                "--manifest",
                str(args.manifest),
                "--lock",
                str(args.lock),
            ]
            print(f"START {run_id}", flush=True)
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
                    f"E01 run {run_id} exited {completed.returncode}; audit {log}"
                )
            payload = json.loads(receipt.read_text())
            if (
                payload.get("status") != "PASS"
                or sha256_file(checkpoint) != payload.get("checkpoint_sha256")
                or sha256_file(output) != payload["validation"].get("output_sha256")
            ):
                raise RuntimeError(f"E01 terminal validation failed: {run_id}")
            print(f"PASS {run_id}", flush=True)


if __name__ == "__main__":
    main()
