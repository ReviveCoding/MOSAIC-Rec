"""Fresh PixelRec p2 temporal roles, cohort construction, and leakage audit."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import duckdb

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import canonical_json_sha256, sha256_file, write_json_new


def controlled_zero_selected(item_id: str, fraction: float = 0.05) -> bool:
    payload = f"{PROTOCOL_ID}:controlled-zero:{item_id}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return value < math.floor(fraction * (1 << 256))


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _write_lines_new(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(f"{value}\n")


def prepare_pixelrec(
    *, repo: Path, source: Path, item_source: Path, scratch: Path
) -> dict[str, dict[str, Any]]:
    protocol = json.loads((repo / "configs/p2/protocol.json").read_text())
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Protocol identity mismatch")
    expected_source = protocol["pixelrec_source"]["interaction_sha256"]
    source_hash = sha256_file(source)
    if source_hash != expected_source:
        raise RuntimeError("PixelRec source hash mismatch")
    item_hash = sha256_file(item_source)
    if scratch.exists() and any(scratch.iterdir()):
        raise FileExistsError(f"Refusing to overwrite p2 data directory: {scratch}")
    scratch.mkdir(parents=True, exist_ok=True)

    split = protocol["temporal_split"]
    fit_end = int(split["fit_max_timestamp_inclusive"])
    val_end = int(split["validation_max_timestamp_inclusive"])
    qual_end = int(split["qualification_max_timestamp_inclusive"])
    max_end = int(split["locked_final_max_timestamp_inclusive"])
    if not fit_end < val_end < qual_end < max_end:
        raise ValueError("Invalid p2 temporal boundaries")

    con = duckdb.connect()
    con.execute("SET threads=4")
    relation = f"read_csv_auto('{_sql_path(source)}', header=true)"
    item_relation = f"read_csv_auto('{_sql_path(item_source)}', header=true)"
    qa = con.execute(
        f"""
        SELECT count(*) AS row_count,
               count(DISTINCT user_id) AS user_count,
               count(DISTINCT item_id) AS item_count,
               min(timestamp) AS min_timestamp,
               max(timestamp) AS max_timestamp,
               count(*) FILTER (WHERE user_id IS NULL OR item_id IS NULL OR timestamp IS NULL) AS null_keys,
               count(*) - count(DISTINCT (user_id, item_id, timestamp)) AS duplicate_events
        FROM {relation}
        """
    ).fetchone()
    schema = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    item_qa = con.execute(
        f"""
        SELECT count(*) AS row_count, count(DISTINCT item_id) AS item_count,
               count(*) FILTER (WHERE item_id IS NULL) AS null_item_ids,
               count(*) FILTER (WHERE title IS NULL OR trim(title)='') AS empty_titles,
               count(*) FILTER (WHERE tag IS NULL OR trim(tag)='') AS empty_tags,
               count(*) FILTER (WHERE description IS NULL OR trim(description)='') AS empty_descriptions
        FROM {item_relation}
        """
    ).fetchone()
    if qa[0] != 989494 or qa[5] or qa[6]:
        raise RuntimeError(f"PixelRec source QA failed: {qa}")

    fit_items = sorted(
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT item_id FROM {relation} WHERE timestamp <= ?",
            [fit_end],
        ).fetchall()
    )
    controlled = [item for item in fit_items if controlled_zero_selected(item)]
    _write_lines_new(scratch / "controlled_zero_item_ids.txt", controlled)
    con.execute("CREATE TABLE controlled(item_id VARCHAR PRIMARY KEY)")
    con.executemany(
        "INSERT INTO controlled VALUES (?)", [(item,) for item in controlled]
    )

    projection = (
        "CAST(user_id AS VARCHAR) user_id, CAST(item_id AS VARCHAR) item_id, "
        'CAST(timestamp AS BIGINT) AS "timestamp"'
    )
    predicates = {
        "FIT_VISIBLE": (
            f"timestamp <= {fit_end} AND item_id NOT IN (SELECT item_id FROM controlled)"
        ),
        "FIT_WITHHELD_CONTROLLED": (
            f"timestamp <= {fit_end} AND item_id IN (SELECT item_id FROM controlled)"
        ),
        "VALIDATION": f"timestamp > {fit_end} AND timestamp <= {val_end}",
        "QUALIFICATION": f"timestamp > {val_end} AND timestamp <= {qual_end}",
        "LOCKED_FINAL": f"timestamp > {qual_end} AND timestamp <= {max_end}",
    }
    outputs: dict[str, dict[str, Any]] = {}
    for role, predicate in predicates.items():
        output = scratch / f"{role}.parquet"
        con.execute(
            f"COPY (SELECT {projection} FROM {relation} WHERE {predicate} "
            f"ORDER BY timestamp, user_id, item_id) TO '{_sql_path(output)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(output)}')"
        ).fetchone()[0]
        outputs[role] = {
            "path": str(output),
            "rows": rows,
            "sha256": sha256_file(output),
        }

    first_seen = scratch / "item_first_observed.parquet"
    con.execute(
        f"COPY (SELECT item_id, min(timestamp)::BIGINT first_observed_timestamp "
        f"FROM {relation} GROUP BY item_id ORDER BY item_id) "
        f"TO '{_sql_path(first_seen)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    support = scratch / "fit_item_support.parquet"
    con.execute(
        f"COPY (SELECT item_id, count(*)::BIGINT fit_interactions, "
        f"count(DISTINCT user_id)::BIGINT fit_unique_users FROM "
        f"read_parquet('{_sql_path(scratch / 'FIT_VISIBLE.parquet')}') "
        f"GROUP BY item_id ORDER BY item_id) TO '{_sql_path(support)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    user_stats = scratch / "fit_user_stats.parquet"
    con.execute(
        f"COPY (SELECT user_id, count(*)::BIGINT fit_history_length, "
        f"count(DISTINCT item_id)::BIGINT fit_unique_items, max(timestamp)::BIGINT last_fit_timestamp "
        f"FROM read_parquet('{_sql_path(scratch / 'FIT_VISIBLE.parquet')}') "
        f"GROUP BY user_id ORDER BY user_id) TO '{_sql_path(user_stats)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    natural = sorted(
        row[0]
        for row in con.execute(
            f"""
            SELECT DISTINCT s.item_id
            FROM {relation} s
            LEFT JOIN read_parquet('{_sql_path(support)}') f USING(item_id)
            LEFT JOIN controlled c USING(item_id)
            WHERE f.item_id IS NULL AND c.item_id IS NULL
            """
        ).fetchall()
    )
    _write_lines_new(scratch / "natural_zero_item_ids.txt", natural)

    auxiliary = {
        "controlled_zero_item_ids": scratch / "controlled_zero_item_ids.txt",
        "natural_zero_item_ids": scratch / "natural_zero_item_ids.txt",
        "item_first_observed": first_seen,
        "fit_item_support": support,
        "fit_user_stats": user_stats,
    }
    for name, path in auxiliary.items():
        outputs[name] = {"path": str(path), "sha256": sha256_file(path)}

    total = sum(outputs[role]["rows"] for role in predicates)
    visible_cold = con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_path(scratch / 'FIT_VISIBLE.parquet')}') "
        "WHERE item_id IN (SELECT item_id FROM controlled)"
    ).fetchone()[0]
    natural_with_support = con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_path(support)}') "
        f"WHERE item_id IN (SELECT * FROM read_csv('{_sql_path(scratch / 'natural_zero_item_ids.txt')}', "
        "columns={'item_id':'VARCHAR'}, header=false))"
    ).fetchone()[0]
    boundary_violations = {
        "FIT_VISIBLE": con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(scratch / 'FIT_VISIBLE.parquet')}') WHERE timestamp > {fit_end}"
        ).fetchone()[0],
        "FIT_WITHHELD_CONTROLLED": con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(scratch / 'FIT_WITHHELD_CONTROLLED.parquet')}') WHERE timestamp > {fit_end}"
        ).fetchone()[0],
        "VALIDATION": con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(scratch / 'VALIDATION.parquet')}') WHERE timestamp <= {fit_end} OR timestamp > {val_end}"
        ).fetchone()[0],
        "QUALIFICATION": con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(scratch / 'QUALIFICATION.parquet')}') WHERE timestamp <= {val_end} OR timestamp > {qual_end}"
        ).fetchone()[0],
        "LOCKED_FINAL": con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(scratch / 'LOCKED_FINAL.parquet')}') WHERE timestamp <= {qual_end} OR timestamp > {max_end}"
        ).fetchone()[0],
    }
    role_population: dict[str, dict[str, int]] = {}
    for role in ("VALIDATION", "QUALIFICATION", "LOCKED_FINAL"):
        row = con.execute(
            f"""
            WITH eligible AS (
              SELECT r.*, coalesce(s.fit_history_length, 0) fit_history_length,
                     coalesce(i.fit_interactions, 0) fit_support,
                     row_number() OVER (PARTITION BY r.user_id ORDER BY r.timestamp, r.item_id) rn
              FROM read_parquet('{_sql_path(scratch / f"{role}.parquet")}') r
              LEFT JOIN read_parquet('{_sql_path(user_stats)}') s USING(user_id)
              LEFT JOIN read_parquet('{_sql_path(support)}') i USING(item_id)
              WHERE coalesce(s.fit_history_length, 0) >= 5
            )
            SELECT count(*) FILTER (WHERE rn=1),
                   count(*) FILTER (WHERE rn=1 AND item_id IN (SELECT item_id FROM controlled)),
                   count(*) FILTER (WHERE rn=1 AND fit_support=0 AND item_id NOT IN (SELECT item_id FROM controlled)),
                   count(*) FILTER (WHERE rn=1 AND fit_support>20)
            FROM eligible
            """
        ).fetchone()
        role_population[role] = {
            "eligible_users": row[0],
            "controlled_zero_targets": row[1],
            "natural_zero_targets": row[2],
            "warm_support_gt_20_targets": row[3],
        }

    leakage_pass = (
        total == qa[0]
        and visible_cold == 0
        and natural_with_support == 0
        and not any(boundary_violations.values())
    )
    if not leakage_pass:
        raise RuntimeError("p2 mechanical leakage audit failed")

    common = {
        "protocol_id": PROTOCOL_ID,
        "git_commit": _git_head(repo),
        "source_sha256": source_hash,
        "item_source_sha256": item_hash,
        "protocol_config_sha256": sha256_file(repo / "configs/p2/protocol.json"),
    }
    receipts = {
        "P2-A02": {
            **common,
            "stage_id": "P2-A02",
            "status": "PASS",
            "dataset": "pixelrec50k",
            "source_path": str(source),
            "item_source_path": str(item_source),
            "official_source": "https://github.com/westlake-repl/PixelRec",
            "evidence_class": "p2_measured_provenance",
        },
        "P2-A03": {
            **common,
            "stage_id": "P2-A03",
            "status": "PASS",
            "interaction_schema": schema,
            "interaction_rows": qa[0],
            "users": qa[1],
            "items": qa[2],
            "min_timestamp": qa[3],
            "max_timestamp": qa[4],
            "null_keys": qa[5],
            "duplicate_events": qa[6],
            "item_rows": item_qa[0],
            "item_ids": item_qa[1],
            "null_item_ids": item_qa[2],
            "empty_titles": item_qa[3],
            "empty_tags": item_qa[4],
            "empty_descriptions": item_qa[5],
            "evidence_class": "p2_measured_schema_qa",
        },
        "P2-A04": {
            **common,
            "stage_id": "P2-A04",
            "status": "PASS",
            "source_population": {"rows": qa[0], "users": qa[1], "items": qa[2]},
            "role_population": role_population,
            "controlled_zero_items": len(controlled),
            "natural_zero_items": len(natural),
            "availability_proxy": "first_observed_interaction_timestamp",
            "evidence_class": "p2_descriptive_eda",
        },
        "P2-A05": {
            **common,
            "stage_id": "P2-A05",
            "status": "PASS",
            "boundaries": split,
            "controlled_zero_method": protocol["controlled_zero_support"],
            "controlled_zero_count": len(controlled),
            "natural_zero_count": len(natural),
            "outputs": outputs,
            "output_manifest_sha256": canonical_json_sha256(outputs),
            "evidence_class": "p2_derived_temporal_protocol",
        },
        "P2-A06": {
            **common,
            "stage_id": "P2-A06",
            "status": "PASS",
            "source_rows": qa[0],
            "split_rows": total,
            "boundary_violations": boundary_violations,
            "controlled_items_with_visible_fit_rows": visible_cold,
            "natural_zero_items_with_fit_support": natural_with_support,
            "role_population": role_population,
            "candidate_time_test": "first_observed_timestamp <= query timestamp required by evaluator contract",
            "warnings": [
                "First-observed time is an availability proxy, not launch time.",
                "Untimestamped content revisions cannot be mechanically ruled out.",
            ],
            "evidence_class": "p2_measured_mechanical_leakage_audit",
        },
    }
    for stage, receipt in receipts.items():
        write_json_new(repo / f"evidence/p2/{stage.lower()}_pixelrec.json", receipt)
    return receipts
