"""Summarize frozen E01 independent-seed qualification without selecting winners."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/e01_seed_qualification_p1.json"
OUTPUT = ROOT / "evidence/e01_seed_qualification_p1.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidates(manifest: dict) -> list[dict]:
    return [
        manifest["sequence_candidate"],
        manifest["cold_baseline_candidate"],
        *manifest["mosaic_candidates"],
    ]


def value(receipt: dict, slice_name: str) -> float:
    slices = receipt.get("validation_slices") or {}
    record = slices.get(slice_name) or {}
    metric = record.get("ndcg_at_20")

    if metric is None:
        raise ValueError(
            f"Missing {slice_name} NDCG@20 in "
            f"{receipt.get('config_id')} seed {receipt.get('seed')}"
        )

    return float(metric)


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_ndcg_at_20": statistics.mean(values),
        "sample_sd_ndcg_at_20": (statistics.stdev(values) if len(values) > 1 else 0.0),
        "min_ndcg_at_20": min(values),
        "max_ndcg_at_20": max(values),
    }


def effective_receipt_path(base: Path) -> Path:
    """Use a valid distinct attempt-2 pass after a preserved attempt-1 failure."""
    receipt = json.loads(base.read_text())
    if receipt.get("status") == "E01_SEED_PASS":
        return base
    retry = base.with_name(f"{base.stem}_attempt2.json")
    if not retry.is_file():
        return base
    retried = json.loads(retry.read_text())
    if (
        retried.get("status") != "E01_SEED_PASS"
        or retried.get("attempt") != 2
        or retried.get("retry_of_receipt") != str(base.relative_to(ROOT))
        or retried.get("retry_of_receipt_sha256") != digest(base)
    ):
        raise ValueError(f"Invalid E01 infrastructure retry: {retry}")
    return retry


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    manifest = json.loads(MANIFEST.read_text())
    manifest_hash = digest(MANIFEST)

    seeds = [int(seed) for seed in manifest["independent_seed_policy"]["seeds"]]

    result_candidates: dict[str, dict] = {}

    for candidate in candidates(manifest):
        config_id = candidate["config_id"]
        slug = config_id.lower()

        receipts = []

        for seed in seeds:
            path = ROOT / "evidence" / f"e01_seedqual_p1_{slug}_seed{seed}.json"

            if not path.is_file():
                raise FileNotFoundError(path)

            path = effective_receipt_path(path)

            receipt = json.loads(path.read_text())

            if (
                receipt.get("stage_id") != "E01"
                or receipt.get("config_id") != config_id
                or receipt.get("seed") != seed
                or receipt.get("status") != "E01_SEED_PASS"
                or receipt.get("e01_manifest_sha256") != manifest_hash
            ):
                raise ValueError(f"E01 receipt identity/status mismatch: {path}")

            receipts.append(
                {
                    "receipt_id": path.stem,
                    "receipt_sha256": digest(path),
                    "seed": seed,
                    "checkpoint_sha256": receipt.get("checkpoint_sha256"),
                    "all_ndcg_at_20": value(receipt, "all"),
                    "controlled_cold_ndcg_at_20": value(
                        receipt,
                        "controlled_cold",
                    ),
                    "warm_fit_ndcg_at_20": value(
                        receipt,
                        "warm_fit",
                    ),
                    "elapsed_wall_seconds": receipt.get("elapsed_wall_seconds"),
                }
            )

        slices = {}

        for slice_name, key in (
            ("all", "all_ndcg_at_20"),
            ("controlled_cold", "controlled_cold_ndcg_at_20"),
            ("warm_fit", "warm_fit_ndcg_at_20"),
        ):
            slices[slice_name] = stats([float(receipt[key]) for receipt in receipts])

        result_candidates[config_id] = {
            "source_hpo_receipt": candidate["source_receipt"],
            "learning_rate": candidate["learning_rate"],
            "epochs": candidate["epochs"],
            "seed_receipts": receipts,
            "slices": slices,
        }

    result = {
        "stage_id": "E01",
        "protocol_id": manifest["protocol_id"],
        "state": "DEVELOPMENT",
        "status": "E01_SEED_QUALIFICATION_COMPLETE_UNQUALIFIED",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "source_hpo_closure_commit": manifest["source_hpo_closure_commit"],
        "seeds": seeds,
        "candidate_count": len(result_candidates),
        "expected_seed_runs": len(result_candidates) * len(seeds),
        "candidates": result_candidates,
        "warnings": [
            "E01 uses VALIDATION only.",
            "HPO seed 20260920 is excluded from E01 statistics.",
            "No comparator or MOSAIC finalist is frozen here.",
            "QUALIFICATION and LOCKED_FINAL remain unopened.",
        ],
        "evidence_class": "derived_development_seed_qualification",
    }

    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
