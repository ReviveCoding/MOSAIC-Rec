"""Mechanical temporal and controlled-cold split leakage checks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def audit(dataset: str, split_dir: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    bounds = config["datasets"][dataset]
    con = duckdb.connect()
    roles = ["FIT_VISIBLE", "FIT_WITHHELD_COLD", "VALIDATION", "QUALIFICATION", "LOCKED_FINAL"]
    paths = {role: split_dir / f"{role}.parquet" for role in roles}
    for role, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        con.execute(f"CREATE VIEW {role} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    violating_bounds = {
        "fit_visible": con.execute(f"SELECT count(*) FROM FIT_VISIBLE WHERE timestamp > {bounds['fit_end']}").fetchone()[0],
        "fit_withheld": con.execute(f"SELECT count(*) FROM FIT_WITHHELD_COLD WHERE timestamp > {bounds['fit_end']}").fetchone()[0],
        "validation": con.execute(f"SELECT count(*) FROM VALIDATION WHERE timestamp <= {bounds['fit_end']} OR timestamp > {bounds['validation_end']}").fetchone()[0],
        "qualification": con.execute(f"SELECT count(*) FROM QUALIFICATION WHERE timestamp <= {bounds['validation_end']} OR timestamp > {bounds['qualification_end']}").fetchone()[0],
        "locked_final": con.execute(f"SELECT count(*) FROM LOCKED_FINAL WHERE timestamp <= {bounds['qualification_end']}").fetchone()[0],
    }
    cohort_path = split_dir / "controlled_cold_item_ids.txt"
    cold_ids = set(cohort_path.read_text().splitlines())
    visible_cold = con.execute("SELECT DISTINCT item_id FROM FIT_VISIBLE").fetchall()
    visible_cold_count = sum(row[0] in cold_ids for row in visible_cold)
    withheld_items = {row[0] for row in con.execute("SELECT DISTINCT item_id FROM FIT_WITHHELD_COLD").fetchall()}
    withheld_outside_cohort = len(withheld_items - cold_ids)
    all_roles = " UNION ALL ".join(
        f"SELECT user_id, item_id, timestamp FROM {role}" for role in roles
    )
    duplicate_events = con.execute(
        f"SELECT count(*) - count(DISTINCT (user_id,item_id,timestamp)) FROM ({all_roles})"
    ).fetchone()[0]
    pass_audit = not (any(violating_bounds.values()) or visible_cold_count or
                      withheld_outside_cohort or duplicate_events)
    return {
        "stage_id": "A06", "protocol_id": config["protocol_id"],
        "state": "DEVELOPMENT", "status": "PASS" if pass_audit else "FAIL",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "temporal_boundary_violations": violating_bounds,
        "controlled_cold_items_with_visible_fit_interactions": visible_cold_count,
        "withheld_items_outside_cold_cohort": withheld_outside_cohort,
        "cross_role_duplicate_events": duplicate_events,
        "warnings": [
            "This mechanical audit cannot establish historical validity of untimestamped content revisions.",
            "Candidate-time filtering is enforced by evaluators and requires its own test.",
        ],
        "evidence_class": "measured_leakage_audit",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["microlens-100k", "pixelrec50k", "ml-32m"])
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset, args.split_dir, args.config)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
