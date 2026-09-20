"""Exact FIT-only popularity baseline on protocol validation roles."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path

import duckdb
import pyarrow as pa


def evaluate(dataset: str, split_dir: Path, half_life_days: float | None = None) -> dict:
    if half_life_days is not None and half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    fit_path = split_dir / "FIT_VISIBLE.parquet"
    withheld_path = split_dir / "FIT_WITHHELD_COLD.parquet"
    val_path = split_dir / "VALIDATION.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW fit AS SELECT * FROM read_parquet('{fit_path.as_posix()}')")
    con.execute(f"CREATE VIEW withheld AS SELECT * FROM read_parquet('{withheld_path.as_posix()}')")
    con.execute(f"CREATE VIEW val AS SELECT * FROM read_parquet('{val_path.as_posix()}')")
    fit_counts = {item: count for item, count in con.execute(
        "SELECT item_id, count(*) FROM fit GROUP BY item_id").fetchall()}
    if half_life_days is None:
        ranking_scores = fit_counts
    else:
        fit_end = con.execute("SELECT max(timestamp) FROM fit").fetchone()[0]
        time_unit = 1000 if dataset == "microlens-100k" else 1
        half_life_units = half_life_days * 86400 * time_unit
        ranking_scores = {item: score for item, score in con.execute(
            "SELECT item_id, sum(exp(ln(2.0) * (timestamp - ?) / ?)) "
            "FROM fit GROUP BY item_id", [fit_end, half_life_units]
        ).fetchall()}
    first_seen = {item: stamp for item, stamp in con.execute(
        "SELECT item_id, min(timestamp) FROM (SELECT item_id,timestamp FROM fit "
        "UNION ALL SELECT item_id,timestamp FROM withheld "
        "UNION ALL SELECT item_id,timestamp FROM val) GROUP BY item_id").fetchall()}
    first_val = con.execute(
        "SELECT user_id,item_id,timestamp FROM (SELECT *, row_number() OVER "
        "(PARTITION BY user_id ORDER BY timestamp,item_id) rn FROM val) WHERE rn=1"
    ).fetchall()
    eval_users = sorted({row[0] for row in first_val})
    con.register("eval_users", pa.table({"user_id": pa.array(eval_users, type=pa.string())}))
    seen: dict[str, set[str]] = defaultdict(set)
    for user, item in con.execute(
        "SELECT fit.user_id, fit.item_id FROM fit JOIN eval_users USING (user_id)"
    ).fetchall():
        seen[user].add(item)
    ranking = sorted(first_seen, key=lambda item: (-ranking_scores.get(item, 0), item))
    cohort_path = split_dir / "controlled_cold_item_ids.txt"
    cohort = set(cohort_path.read_text().splitlines()) if cohort_path.exists() else set()
    slices = {"all": [0.0, 0.0, 0.0, 0], "controlled_cold": [0.0, 0.0, 0.0, 0],
              "warm_fit": [0.0, 0.0, 0.0, 0]}
    excluded_no_fit = 0
    for user, target, stamp in first_val:
        if user not in seen:
            excluded_no_fit += 1
            continue
        history = seen[user]
        top = []
        for item in ranking:
            if first_seen[item] <= stamp and item not in history:
                top.append(item)
                if len(top) == 20:
                    break
        labels = ["all"]
        if target in cohort:
            labels.append("controlled_cold")
        elif target in fit_counts:
            labels.append("warm_fit")
        for label in labels:
            row = slices[label]
            row[3] += 1
            if target in top:
                rank = top.index(target) + 1
                row[0] += 1 / log2(rank + 1)
                row[1] += 1
                row[2] += 1 / rank
    result_slices = {
        label: {"n": int(v[3]), "ndcg_at_20": v[0] / v[3] if v[3] else None,
                "recall_at_20": v[1] / v[3] if v[3] else None,
                "mrr_at_20": v[2] / v[3] if v[3] else None}
        for label, v in slices.items()
    }
    return {
        "stage_id": "C00", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT", "status": "PILOT_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "model": "B0_GLOBAL_FIT_POPULARITY" if half_life_days is None else "B1_RECENT_FIT_POPULARITY",
        "half_life_days": half_life_days,
        "role": "VALIDATION", "validation_slices": result_slices,
        "excluded_no_fit_user": excluded_no_fit,
        "ranking": "Exact all available FIT-or-validation items, FIT-only "
                   + ("counts" if half_life_days is None else "exponentially decayed counts at FIT end")
                   + ", item-ID tie break",
        "candidate_availability": "first observed timestamp <= event timestamp",
        "warnings": ["Validation pilot only; no locked result or paired inferential claim.",
                     "First observed time is a visibility proxy, not launch time."],
        "evidence_class": "measured_development_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["microlens-100k", "pixelrec50k", "ml-32m"])
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--half-life-days", type=float)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.split_dir, args.half_life_days)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
