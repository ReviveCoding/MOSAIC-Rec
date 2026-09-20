"""Select a declared deterministic B1 development setting from validation receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LABELS = {"microlens-100k": "microlens", "pixelrec50k": "pixelrec", "ml-32m": "ml32m"}


def select(config_path: Path, evidence: Path) -> dict:
    config = json.loads(config_path.read_text())
    if (
        config["selection_role"] != "VALIDATION"
        or config["selection_metric"] != "all_ndcg_at_20"
    ):
        raise ValueError("Unexpected B1 selection contract")
    trials: dict[str, list[dict]] = {}
    choices: dict[str, dict] = {}
    for dataset, label in LABELS.items():
        trial_rows = []
        user_counts = set()
        for days in config["half_life_days"]:
            path = evidence / f"{label}_recent_popularity_{days}d_pilot.json"
            data = json.loads(path.read_text())
            if (
                data["dataset"] != dataset
                or data["model"] != config["model"]
                or data["role"] != "VALIDATION"
                or data["half_life_days"] != days
            ):
                raise ValueError(f"Receipt contract mismatch: {path}")
            metric = data["validation_slices"]["all"]
            user_counts.add(metric["n"])
            trial_rows.append(
                {
                    "half_life_days": days,
                    "receipt_id": path.stem,
                    "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "all_ndcg_at_20": metric["ndcg_at_20"],
                    "users": metric["n"],
                    "cold_ndcg_at_20": data["validation_slices"]["controlled_cold"][
                        "ndcg_at_20"
                    ],
                    "warm_ndcg_at_20": data["validation_slices"]["warm_fit"][
                        "ndcg_at_20"
                    ],
                }
            )
        if len(user_counts) != 1:
            raise ValueError(f"Inconsistent validation populations for {dataset}")
        selected = max(
            trial_rows, key=lambda row: (row["all_ndcg_at_20"], row["half_life_days"])
        )
        trials[dataset] = trial_rows
        choices[dataset] = {
            "half_life_days": selected["half_life_days"],
            "receipt_id": selected["receipt_id"],
            "all_ndcg_at_20": selected["all_ndcg_at_20"],
        }
    return {
        "stage_id": "C00",
        "protocol_id": config["protocol_id"],
        "state": "DEVELOPMENT",
        "status": "VALIDATION_SELECTED",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "trials": trials,
        "selected": choices,
        "warnings": [
            "Validation-selected descriptive baseline, not a frozen neural comparator or locked result."
        ],
        "evidence_class": "derived_development_selection",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = select(args.config, args.evidence_dir)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
