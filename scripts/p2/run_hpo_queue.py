"""Run the frozen p2 pre-router HPO queue serially and crash safely."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mosaic_rec.p2.contracts import sha256_file
from mosaic_rec.p2.hpo_configs import validate_trial_identity

FAMILY_ORDER = (
    "CSEQ_BERT4REC",
    "CCOLD_TWO_TOWER",
    "LEGACY_M1",
    "BPR_MF",
    "LIGHTGCN",
    "SASREC",
    "VBPR",
)

AUDITED_LAUNCH_ATTEMPTS = {
    "LIGHTGCN_T3": 2,
}


def trial_paths(repo: Path, scratch: Path, trial_id: str) -> dict[str, Path]:
    stem = trial_id.lower()
    launch_attempt = AUDITED_LAUNCH_ATTEMPTS.get(trial_id, 1)
    log_suffix = "" if launch_attempt == 1 else f"_attempt{launch_attempt}"
    return {
        "config": repo / f"configs/p2/hpo_trials/{stem}.json",
        "checkpoint": scratch / f"hpo/checkpoints/{stem}.pt",
        "output": scratch / f"hpo/results/{stem}_validation.parquet",
        "receipt": repo / f"evidence/p2/p2-e00-{stem}.json",
        "log": scratch / f"hpo/logs/{stem}{log_suffix}.log",
    }


def validate_terminal(paths: dict[str, Path], config_hash: str) -> dict[str, Any]:
    receipt = json.loads(paths["receipt"].read_text())
    if receipt.get("status") != "PASS":
        raise RuntimeError(f"Existing receipt is not terminal PASS: {paths['receipt']}")
    if receipt.get("config_sha256") != config_hash:
        raise RuntimeError(f"Config hash mismatch: {paths['receipt']}")
    for field, path_key in (("checkpoint_sha256", "checkpoint"),):
        path = paths[path_key]
        if not path.is_file() or receipt.get(field) != sha256_file(path):
            raise RuntimeError(f"Artifact hash mismatch: {path}")
    validation = receipt.get("validation", {})
    output = paths["output"]
    if not output.is_file() or validation.get("output_sha256") != sha256_file(output):
        raise RuntimeError(f"Validation output hash mismatch: {output}")
    return receipt


def manifest_config_hash(identity: dict[str, Any]) -> str:
    """Return the byte-level trial-config hash stored by materialization."""
    value = identity.get("sha256")
    if not isinstance(value, str) or not value:
        raise RuntimeError("HPO manifest identity is missing sha256")
    return value


def command_for(
    *, family: str, trial_id: str, repo: Path, scratch: Path, lock: Path
) -> list[str]:
    if family in {"CSEQ_BERT4REC", "CCOLD_TWO_TOWER"}:
        script = repo / "scripts/p2/run_expert_hpo_trial.py"
    elif family == "LEGACY_M1":
        script = repo / "scripts/p2/run_legacy_hpo_trial.py"
    else:
        script = repo / "scripts/p2/run_baseline_hpo_trial.py"
    return [
        sys.executable,
        str(script),
        trial_id,
        "--repo",
        str(repo),
        "--scratch",
        str(scratch),
        "--lock",
        str(lock),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    scratch = args.scratch.resolve()
    hpo_path = repo / "configs/p2/hpo.json"
    manifest_path = repo / "configs/p2/hpo_trials_manifest.json"
    hpo = json.loads(hpo_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected_ids = [
        trial["trial_id"]
        for family in FAMILY_ORDER
        for trial in hpo["families"][family]
    ]
    for trial_id in expected_ids:
        paths = trial_paths(repo, scratch, trial_id)
        identity = validate_trial_identity(
            manifest=manifest,
            trial_id=trial_id,
            config_path=paths["config"],
            hpo_path=hpo_path,
        )
        if paths["receipt"].exists():
            validate_terminal(paths, manifest_config_hash(identity))
            print(f"SKIP_TERMINAL {trial_id}", flush=True)
            continue
        partials = [
            str(paths[key])
            for key in ("checkpoint", "output", "log")
            if paths[key].exists()
        ]
        if partials:
            raise RuntimeError(
                f"Audit partial artifacts before retrying {trial_id}: {partials}"
            )
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        source = str(repo / "src")
        environment["PYTHONPATH"] = (
            source
            if not environment.get("PYTHONPATH")
            else source + os.pathsep + environment["PYTHONPATH"]
        )
        print(f"START {trial_id}", flush=True)
        with paths["log"].open("xb") as log:
            completed = subprocess.run(
                command_for(
                    family=identity["family"],
                    trial_id=trial_id,
                    repo=repo,
                    scratch=scratch,
                    lock=args.lock,
                ),
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Trial {trial_id} exited {completed.returncode}; audit {paths['log']}"
            )
        validate_terminal(paths, manifest_config_hash(identity))
        print(f"PASS {trial_id}", flush=True)


if __name__ == "__main__":
    main()
