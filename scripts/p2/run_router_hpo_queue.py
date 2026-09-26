"""Run the frozen p2 router HPO trials serially and crash safely."""

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
    manifest_path = repo / "configs/p2/router_hpo_trials_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    for identity in manifest["trials"]:
        trial_id = identity["trial_id"]
        stem = trial_id.lower()
        config = repo / identity["path"]
        checkpoint = scratch / f"hpo/checkpoints/{stem}.pt"
        output = scratch / f"hpo/results/{stem}_validation.parquet"
        receipt = repo / f"evidence/p2/p2-e00-{stem}.json"
        log = scratch / f"hpo/logs/{stem}.log"
        if sha256_file(config) != identity["sha256"]:
            raise RuntimeError(f"Router config hash mismatch: {trial_id}")
        if receipt.exists():
            payload = json.loads(receipt.read_text())
            if (
                payload.get("status") != "PASS"
                or payload.get("config_sha256") != identity["sha256"]
                or not checkpoint.is_file()
                or payload.get("checkpoint_sha256") != sha256_file(checkpoint)
                or not output.is_file()
                or payload.get("validation", {}).get("output_sha256")
                != sha256_file(output)
            ):
                raise RuntimeError(f"Invalid terminal router trial: {trial_id}")
            print(f"SKIP_TERMINAL {trial_id}", flush=True)
            continue
        partials = [str(path) for path in (checkpoint, output, log) if path.exists()]
        if partials:
            raise RuntimeError(f"Audit router partials for {trial_id}: {partials}")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(repo / "scripts/p2/run_router_hpo_trial.py"),
            trial_id,
            "--repo",
            str(repo),
            "--scratch",
            str(scratch),
            "--manifest",
            str(manifest_path),
            "--lock",
            str(args.lock),
        ]
        print(f"START {trial_id}", flush=True)
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
            raise RuntimeError(f"Router trial {trial_id} exited {completed.returncode}")
        print(f"PASS {trial_id}", flush=True)


if __name__ == "__main__":
    main()
