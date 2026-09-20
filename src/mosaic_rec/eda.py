"""Pre-freeze descriptive analysis of official interaction files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def inspect(dataset: str, path: Path) -> dict:
    con = duckdb.connect()
    relation = f"read_csv('{path.as_posix()}', header=true, auto_detect=true)"
    user, item, stamp = {
        "microlens-100k": ("user", "item", "timestamp"),
        "ml-32m": ("userId", "movieId", "timestamp"),
        "pixelrec50k": ("user_id", "item_id", "timestamp"),
    }[dataset]
    cutoffs = con.execute(
        f"SELECT quantile_disc({stamp}, 0.7), quantile_disc({stamp}, 0.8), "
        f"quantile_disc({stamp}, 0.9) FROM {relation}"
    ).fetchone()
    roles = con.execute(
        f"""SELECT CASE WHEN {stamp} <= {cutoffs[0]} THEN 'FIT'
          WHEN {stamp} <= {cutoffs[1]} THEN 'VALIDATION'
          WHEN {stamp} <= {cutoffs[2]} THEN 'QUALIFICATION'
          ELSE 'LOCKED_FINAL' END AS split_role, count(*) events,
          count(DISTINCT {user}) users, count(DISTINCT {item}) items
          FROM {relation} GROUP BY 1 ORDER BY 1"""
    ).fetchall()
    fit_history = con.execute(
        f"""SELECT min(n), approx_quantile(n, [0.25, 0.5, 0.75, 0.95]), max(n)
        FROM (SELECT count(*) n FROM {relation} WHERE {stamp} <= {cutoffs[0]}
        GROUP BY {user})"""
    ).fetchone()
    support = con.execute(
        f"""SELECT count(*) FILTER (WHERE n=1),
        count(*) FILTER (WHERE n BETWEEN 2 AND 5),
        count(*) FILTER (WHERE n BETWEEN 6 AND 20),
        count(*) FILTER (WHERE n>20)
        FROM (SELECT count(*) n FROM {relation} WHERE {stamp} <= {cutoffs[0]}
        GROUP BY {item})"""
    ).fetchone()
    overlap = con.execute(
        f"""WITH per_user AS (
        SELECT {user} AS user_key,
          bool_or({stamp} <= {cutoffs[0]}) AS has_fit,
          bool_or({stamp} > {cutoffs[0]} AND {stamp} <= {cutoffs[1]}) AS has_validation,
          bool_or({stamp} > {cutoffs[1]} AND {stamp} <= {cutoffs[2]}) AS has_qualification,
          bool_or({stamp} > {cutoffs[2]}) AS has_locked
        FROM {relation} GROUP BY 1)
        SELECT count(*) FILTER (WHERE has_fit AND has_validation),
          count(*) FILTER (WHERE has_fit AND has_qualification),
          count(*) FILTER (WHERE has_fit AND has_locked),
          count(*) FILTER (WHERE has_validation AND NOT has_fit),
          count(*) FILTER (WHERE has_locked AND NOT has_fit)
        FROM per_user"""
    ).fetchone()
    return {
        "stage_id": "A04", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "PILOT", "status": "PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "source_file": path.name,
        "exact_candidate_cutoffs_unfrozen": cutoffs,
        "roles_at_candidate_cutoffs": [
            dict(zip(("role", "events", "users", "items"), row)) for row in roles
        ],
        "fit_events_per_user_min_quantiles_max": fit_history,
        "fit_item_support_1_2to5_6to20_gt20": support,
        "user_overlap_fit_with_validation_qualification_locked_then_no_fit_validation_locked": overlap,
        "warnings": [
            "Cutoffs are exploratory until split and leakage protocol freeze.",
            "Zero-support items outside FIT are not included in FIT support counts.",
        ],
        "evidence_class": "measured_eda",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["microlens-100k", "ml-32m", "pixelrec50k"])
    parser.add_argument("csv", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = inspect(args.dataset, args.csv)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
