"""FIT-only item/user context and common p2 evaluation query manifests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def build_contexts(
    *, repo: Path, data_dir: Path, feature_dir: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite p2 contexts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        data_dir / "FIT_VISIBLE.parquet",
        data_dir / "VALIDATION.parquet",
        data_dir / "QUALIFICATION.parquet",
        data_dir / "LOCKED_FINAL.parquet",
        data_dir / "fit_item_support.parquet",
        data_dir / "item_first_observed.parquet",
        data_dir / "controlled_zero_item_ids.txt",
        feature_dir / "publisher_image_item_ids.txt",
        feature_dir / "tag_ids.npy",
        feature_dir / "image_present.npy",
        feature_dir / "text_present.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    import numpy as np

    item_ids = (feature_dir / "publisher_image_item_ids.txt").read_text().splitlines()
    tag_ids = np.load(feature_dir / "tag_ids.npy")
    image_present = np.load(feature_dir / "image_present.npy")
    text_present = np.load(feature_dir / "text_present.npy")
    if not (len(item_ids) == len(tag_ids) == len(image_present) == len(text_present)):
        raise RuntimeError("p2 feature context lengths differ")
    item_arrow = pa.table(
        {
            "item_id": item_ids,
            "tag_id": tag_ids,
            "image_present": image_present,
            "text_present": text_present,
        }
    )
    controlled = (data_dir / "controlled_zero_item_ids.txt").read_text().splitlines()
    controlled_arrow = pa.table({"item_id": controlled})

    con = duckdb.connect()
    con.execute("SET threads=4")
    con.register("feature_items", item_arrow)
    con.register("controlled_ids", controlled_arrow)
    con.execute(
        f"CREATE VIEW fit AS SELECT * FROM read_parquet('{_sql_path(data_dir / 'FIT_VISIBLE.parquet')}')"
    )
    con.execute(
        f"CREATE VIEW support AS SELECT * FROM read_parquet('{_sql_path(data_dir / 'fit_item_support.parquet')}')"
    )
    con.execute(
        f"CREATE VIEW first_seen AS SELECT * FROM read_parquet('{_sql_path(data_dir / 'item_first_observed.parquet')}')"
    )
    item_context = output_dir / "item_context.parquet"
    con.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT f.item_id, f.tag_id, f.image_present, f.text_present,
                   coalesce(s.fit_interactions, 0)::BIGINT fit_interactions,
                   coalesce(s.fit_unique_users, 0)::BIGINT fit_unique_users,
                   o.first_observed_timestamp,
                   c.item_id IS NOT NULL is_controlled_zero
            FROM feature_items f
            LEFT JOIN support s USING(item_id)
            JOIN first_seen o USING(item_id)
            LEFT JOIN controlled_ids c USING(item_id)
          ), ranked AS (
            SELECT *,
                   ntile(5) OVER (ORDER BY fit_interactions, item_id) interaction_quintile,
                   ntile(10) OVER (ORDER BY fit_interactions, item_id) interaction_decile,
                   ntile(5) OVER (ORDER BY fit_unique_users, item_id) unique_user_quintile,
                   CASE WHEN fit_interactions > 0 THEN
                     ntile(5) OVER (
                       PARTITION BY (fit_interactions > 0)
                       ORDER BY fit_interactions, item_id
                     ) END positive_support_quintile
            FROM base
          )
          SELECT *,
                 CASE interaction_quintile
                   WHEN 1 THEN 'tail' WHEN 5 THEN 'head' ELSE 'torso'
                 END popularity_tier,
                 positive_support_quintile = 1 positive_support_tail
          FROM ranked ORDER BY item_id
        ) TO '{_sql_path(item_context)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    user_context = output_dir / "user_context.parquet"
    con.execute(
        f"""
        COPY (
          WITH tagged AS (
            SELECT f.user_id, f.item_id, f.timestamp, i.fit_interactions,
                   i.positive_support_tail, i.tag_id
            FROM fit f JOIN read_parquet('{_sql_path(item_context)}') i USING(item_id)
          ), tag_counts AS (
            SELECT user_id, tag_id, count(*)::DOUBLE n
            FROM tagged WHERE tag_id >= 0 GROUP BY user_id, tag_id
          ), tag_totals AS (
            SELECT user_id, sum(n) total FROM tag_counts GROUP BY user_id
          ), entropy AS (
            SELECT c.user_id,
                   -sum((c.n/t.total) * ln(c.n/t.total)) history_tag_entropy
            FROM tag_counts c JOIN tag_totals t USING(user_id) GROUP BY c.user_id
          ), base AS (
            SELECT t.user_id,
                   count(*)::BIGINT fit_history_length,
                   count(DISTINCT t.item_id)::BIGINT fit_unique_items,
                   (1.0 - count(DISTINCT t.item_id)::DOUBLE/count(*)) repeat_ratio,
                   max(t.timestamp)::BIGINT last_fit_timestamp,
                   median(t.fit_interactions)::DOUBLE median_history_popularity,
                   avg(t.positive_support_tail::INTEGER)::DOUBLE history_tail_share,
                   coalesce(e.history_tag_entropy, 0.0) history_tag_entropy
            FROM tagged t LEFT JOIN entropy e USING(user_id)
            GROUP BY t.user_id, e.history_tag_entropy
          )
          SELECT *,
                 ntile(4) OVER (ORDER BY fit_history_length, user_id) history_length_quartile,
                 ntile(4) OVER (ORDER BY median_history_popularity, user_id) niche_quartile,
                 ntile(4) OVER (ORDER BY history_tail_share, user_id) tail_share_quartile,
                 ntile(4) OVER (ORDER BY history_tag_entropy, user_id) diversity_quartile
          FROM base ORDER BY user_id
        ) TO '{_sql_path(user_context)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    query_outputs: dict[str, dict[str, Any]] = {}
    query_counts: dict[str, dict[str, int]] = {}
    for role in ("VALIDATION", "QUALIFICATION", "LOCKED_FINAL"):
        role_path = data_dir / f"{role}.parquet"
        query_path = output_dir / f"{role}_QUERIES.parquet"
        con.execute(
            f"""
            COPY (
              WITH eligible AS (
                SELECT r.user_id, r.item_id target_item_id,
                       r.timestamp query_timestamp,
                       u.* EXCLUDE(user_id),
                       i.* EXCLUDE(item_id),
                       greatest(0.0, (r.timestamp-u.last_fit_timestamp)/86400.0) user_recency_days,
                       greatest(0.0, (r.timestamp-i.first_observed_timestamp)/86400.0) item_age_days,
                       row_number() OVER (
                         PARTITION BY r.user_id ORDER BY r.timestamp, r.item_id
                       ) query_number
                FROM read_parquet('{_sql_path(role_path)}') r
                JOIN read_parquet('{_sql_path(user_context)}') u USING(user_id)
                JOIN read_parquet('{_sql_path(item_context)}') i ON r.item_id=i.item_id
                WHERE u.fit_history_length >= 5
                  AND i.first_observed_timestamp <= r.timestamp
              )
              SELECT * EXCLUDE(query_number),
                     fit_interactions = 0 AND NOT is_controlled_zero is_natural_zero,
                     fit_interactions > 20 is_warm
              FROM eligible WHERE query_number=1 ORDER BY user_id
            ) TO '{_sql_path(query_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        counts = con.execute(
            f"""
            SELECT count(*), count(DISTINCT user_id),
                   count(*) FILTER (WHERE is_controlled_zero),
                   count(*) FILTER (WHERE is_natural_zero),
                   count(*) FILTER (WHERE is_warm),
                   count(*) FILTER (WHERE first_observed_timestamp > query_timestamp)
            FROM read_parquet('{_sql_path(query_path)}')
            """
        ).fetchone()
        if counts[0] != counts[1] or counts[5] != 0:
            raise RuntimeError(f"Invalid {role} query manifest: {counts}")
        query_counts[role] = {
            "queries": counts[0],
            "users": counts[1],
            "controlled_zero": counts[2],
            "natural_zero": counts[3],
            "warm": counts[4],
            "candidate_time_violations": counts[5],
        }
        query_outputs[role] = {
            "path": str(query_path),
            "sha256": sha256_file(query_path),
        }

    outputs = {
        "item_context": {
            "path": str(item_context),
            "sha256": sha256_file(item_context),
        },
        "user_context": {
            "path": str(user_context),
            "sha256": sha256_file(user_context),
        },
        **query_outputs,
    }
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-B02",
        "status": "PASS",
        "git_commit": _git_head(repo),
        "data_manifest_sha256": sha256_file(repo / "evidence/p2/p2-a05_pixelrec.json"),
        "feature_manifest_sha256": sha256_file(
            repo / "evidence/p2/p2-b00_pixelrec.json"
        ),
        "candidate_amendment_sha256": sha256_file(
            repo / "configs/p2/amendment_002_candidate_semantics.json"
        ),
        "candidate_availability": "first_observed_timestamp <= query_timestamp",
        "history_mask": "FIT_VISIBLE history except target",
        "query_policy": "earliest eligible event per user per role",
        "query_counts": query_counts,
        "outputs": outputs,
        "protected_model_outcomes_accessed": False,
        "warnings": [
            "QUALIFICATION and LOCKED_FINAL query rows are protocol inputs, not model outcomes.",
            "First-observed item time is an availability proxy.",
        ],
        "evidence_class": "p2_common_candidate_and_context_contract",
    }
    write_json_new(repo / "evidence/p2/p2-b02_contexts.json", receipt)
    return receipt
