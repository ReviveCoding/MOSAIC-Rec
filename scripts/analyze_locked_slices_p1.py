"""Frozen support bins plus descriptive popularity/history locked slices."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SPLIT = Path("/home/bjw-0/mosaic-rec-work/data/microlens-100k/split_p1")
OUTPUT = ROOT / "evidence/f03_f05_locked_slices_p1.json"
RECEIPTS = {
    "C_seq": ROOT / "evidence/f00_locked_p1_b7_bert4rec_t3.json",
    "C_cold": ROOT / "evidence/f00_locked_p1_b4_two_tower_t3.json",
    "MOSAIC": ROOT / "evidence/f00_locked_p1_m1_t2.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(receipt_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    receipt = json.loads(receipt_path.read_text())
    path = Path(receipt["per_user_path"])
    if digest(path) != receipt["per_user_sha256"]:
        raise ValueError("Per-user evidence hash mismatch")
    table = pq.read_table(path)
    return receipt, {
        name: np.asarray(table[name].to_pylist()) for name in table.column_names
    }


def metric(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    return {
        "n": int(mask.sum()),
        "ndcg_at_20": float(values[mask].mean()) if mask.any() else None,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Locked slice receipt exists; preserve it")
    fit = pq.read_table(SPLIT / "FIT_VISIBLE.parquet", columns=["user_id", "item_id"])
    item_counts = Counter(int(value) for value in fit["item_id"].to_pylist())
    user_counts = Counter(int(value) for value in fit["user_id"].to_pylist())
    receipts: dict[str, dict] = {}
    rows: dict[str, dict[str, np.ndarray]] = {}
    for name, path in RECEIPTS.items():
        receipts[name], rows[name] = load(path)
    reference = rows["MOSAIC"]
    for name in ("C_seq", "C_cold"):
        for key in ("user_id", "item_id", "timestamp", "slice"):
            if not np.array_equal(reference[key], rows[name][key]):
                raise ValueError(f"Paired locked rows differ for {name}:{key}")
    target_support = np.asarray(
        [item_counts[int(item)] for item in reference["item_id"]], dtype=np.int64
    )
    history = np.asarray(
        [user_counts[int(user)] for user in reference["user_id"]], dtype=np.int64
    )
    support_masks = {
        "cold_0": target_support == 0,
        "emerging_1_5": (target_support >= 1) & (target_support <= 5),
        "sparse_6_20": (target_support >= 6) & (target_support <= 20),
        "warm_gt_20": target_support > 20,
    }
    quartiles = np.quantile(history, [0.25, 0.5, 0.75], method="nearest").astype(int)
    history_masks = {
        f"q1_lte_{quartiles[0]}": history <= quartiles[0],
        f"q2_{quartiles[0] + 1}_{quartiles[1]}": (
            (history > quartiles[0]) & (history <= quartiles[1])
        ),
        f"q3_{quartiles[1] + 1}_{quartiles[2]}": (
            (history > quartiles[1]) & (history <= quartiles[2])
        ),
        f"q4_gt_{quartiles[2]}": history > quartiles[2],
    }
    models = {}
    for name, row in rows.items():
        values = row["ndcg_at_20"].astype(np.float64)
        models[name] = {
            "support_bins": {
                label: metric(values, mask) for label, mask in support_masks.items()
            },
            "user_history_quartiles_descriptive": {
                label: metric(values, mask) for label, mask in history_masks.items()
            },
        }
    result = {
        "stage_id": "F03_F05",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "ANALYSIS",
        "status": "LOCKED_SLICE_ANALYSIS_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset": "microlens-100k",
        "support_bin_identity": {
            "cold": 0,
            "emerging": [1, 5],
            "sparse": [6, 20],
            "warm_min": 21,
        },
        "history_quartile_boundaries": quartiles.tolist(),
        "models": models,
        "source_receipts": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest(path),
            }
            for name, path in RECEIPTS.items()
        },
        "warnings": [
            "Support bins are frozen confirmatory slices.",
            "User-history quartiles are post-lock descriptive slices and support no confirmatory claim.",
            "FIT interaction count is a support/popularity proxy, not exposure or causal popularity.",
        ],
        "evidence_class": "derived_locked_slice_analysis",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
