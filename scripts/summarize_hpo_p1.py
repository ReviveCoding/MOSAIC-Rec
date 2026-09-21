"""Close the declared development HPO grid without choosing comparators."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/hpo_development_p1.json"
OUTPUT = ROOT / "evidence/hpo_p1_grid_summary.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(receipt: dict, slice_name: str) -> float | None:
    if slice_name == "all" and receipt.get("ndcg_at_20") is not None:
        value = receipt["ndcg_at_20"]
    else:
        value = (
            (receipt.get("validation_slices") or {})
            .get(slice_name, {})
            .get("ndcg_at_20")
        )
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite HPO validation metric")
    return value


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    policy = json.loads(POLICY.read_text())
    policy_hash = digest(POLICY)
    models = [name for family in policy["model_families"].values() for name in family]
    summaries = {}
    failures = 0
    for model in models:
        trials = []
        for index, learning_rate in enumerate(policy["learning_rates"], start=1):
            path = ROOT / "evidence" / f"hpo_p1_{model.lower()}_t{index}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Declared HPO trial missing: {path}")
            receipt = json.loads(path.read_text())
            if (
                receipt["protocol_id"] != policy["protocol_id"]
                or receipt["model"] != model
                or receipt["hpo_trial_index"] != index
                or receipt["learning_rate"] != learning_rate
                or receipt["epochs"] != policy["max_epochs_per_trial"]
                or receipt["hpo_policy_sha256"] != policy_hash
            ):
                raise ValueError(f"HPO trial identity mismatch: {path}")
            if receipt["status"] not in {"HPO_TRIAL_PASS", "HPO_TRIAL_FAIL"}:
                raise ValueError(f"HPO trial lacks terminal status: {path}")
            if receipt["status"] == "HPO_TRIAL_FAIL":
                failures += 1
            trials.append(
                {
                    "receipt_id": path.stem,
                    "receipt_sha256": digest(path),
                    "status": receipt["status"],
                    "learning_rate": learning_rate,
                    "elapsed_wall_seconds": receipt["elapsed_wall_seconds"],
                    "checkpoint_sha256": receipt.get("checkpoint_sha256"),
                    "all_ndcg_at_20": metric(receipt, "all"),
                    "controlled_cold_ndcg_at_20": metric(receipt, "controlled_cold"),
                    "warm_fit_ndcg_at_20": metric(receipt, "warm_fit"),
                }
            )
        wall = sum(float(trial["elapsed_wall_seconds"]) for trial in trials)
        if wall > policy["gpu_time_budget_seconds_per_model"]:
            raise ValueError(f"Declared per-model time budget exceeded: {model}")
        best = {}
        for slice_name, key in (
            ("all", "all_ndcg_at_20"),
            ("controlled_cold", "controlled_cold_ndcg_at_20"),
            ("warm_fit", "warm_fit_ndcg_at_20"),
        ):
            eligible = [
                trial
                for trial in trials
                if trial["status"] == "HPO_TRIAL_PASS" and trial[key] is not None
            ]
            best[slice_name] = (
                max(eligible, key=lambda trial: (trial[key], -trial["learning_rate"]))[
                    "receipt_id"
                ]
                if eligible
                else None
            )
        summaries[model] = {
            "trials": trials,
            "wall_seconds": wall,
            "validation_shortlist_heads": best,
        }
    output = {
        "stage_id": "E00",
        "protocol_id": policy["protocol_id"],
        "state": "DEVELOPMENT",
        "status": "HPO_GRID_COMPLETE_UNQUALIFIED",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": policy_hash,
        "dataset": "microlens-100k",
        "planned_models": len(models),
        "planned_trials": len(models) * policy["trials_per_model"],
        "failed_trials": failures,
        "models": summaries,
        "warnings": [
            "Validation shortlist heads are not frozen comparators or a MOSAIC finalist.",
            "Reference parity, independent seed qualification, and QUALIFICATION remain pending.",
        ],
        "evidence_class": "derived_development_hpo_analysis",
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
