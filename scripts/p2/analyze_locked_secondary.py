"""Read-only preregistered slice and top-k analysis for p2 LOCKED_FINAL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, verify_p1_immutable, write_json_new
from mosaic_rec.p2.runtime import load_catalog
from mosaic_rec.p2.slices import aggregate_slices, validate_joined_output


def _gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=np.float64))
    if values.sum() == 0:
        return 0.0
    n = values.size
    return float((2 * np.dot(np.arange(1, n + 1), values) / values.sum() - n - 1) / n)


def _topk_metrics(
    rows: list[dict[str, Any]],
    item_to_index: dict[str, int],
    fit_counts: np.ndarray,
    tail: np.ndarray,
    eligible: np.ndarray,
    representation: np.ndarray,
) -> dict[str, Any]:
    lists = [[item_to_index[item] for item in row["top20_item_ids"]] for row in rows]
    flat = np.asarray([item for values in lists for item in values], dtype=np.int64)
    exposure = np.bincount(flat, minlength=fit_counts.size).astype(np.float64)
    eligible_exposure = exposure[eligible]
    shares = eligible_exposure / eligible_exposure.sum()
    nonzero = shares[shares > 0]
    novelty = -np.log2((fit_counts + 1) / (fit_counts.sum() + eligible.sum()))
    diversity = []
    for values in lists:
        vectors = representation[np.asarray(values)]
        total = vectors.sum(axis=0)
        cosine_sum = (float(total @ total) - len(values)) / 2.0
        pairs = len(values) * (len(values) - 1) / 2.0
        diversity.append(1.0 - cosine_sum / pairs)
    return {
        "ItemCoverage_at_20": float(
            np.count_nonzero(eligible_exposure) / eligible.sum()
        ),
        "AveragePopularity_at_20": float(np.log1p(fit_counts[flat]).mean()),
        "TailPercentage_at_20": float(tail[flat].mean()),
        "exposure_Gini": _gini(eligible_exposure),
        "exposure_HHI": float(np.square(shares).sum()),
        "exposure_entropy": float(-(nonzero * np.log(nonzero)).sum()),
        "novelty": float(novelty[flat].mean()),
        "intra_list_diversity": float(np.mean(diversity)),
        "personalization": "NOT_AVAILABLE_PROTOCOL_UNDERSPECIFIED_PAIR_SELECTION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries = pq.read_table(
        args.scratch / "contexts/pixelrec50k/LOCKED_FINAL_QUERIES.parquet"
    ).to_pylist()
    query_map = {
        (row["user_id"], row["query_timestamp"], row["target_item_id"]): row
        for row in queries
    }
    paths = {
        "C_seq": args.scratch / "locked_final/cseq_bert4rec_topk.parquet",
        "C_cold": args.scratch / "locked_final/ccold_two_tower_topk.parquet",
        "R4": args.scratch / "locked_final/r4_topk.parquet",
    }
    slice_tables = []
    model_rows: dict[str, list[dict[str, Any]]] = {}
    for model_id, path in paths.items():
        rows = pq.read_table(path).to_pylist()
        joined = []
        for row in rows:
            key = (row["user_id"], row["query_timestamp"], row["target_item_id"])
            joined.append({**query_map[key], **row})
        validate_joined_output(joined)
        model_rows[model_id] = joined
        aggregated = aggregate_slices(joined)
        table = aggregated.append_column(
            "model_id", pa.array([model_id] * len(aggregated))
        )
        slice_tables.append(table)
    slice_table = pa.concat_tables(slice_tables)
    slice_path = args.output_dir / "locked_slice_metrics.parquet"
    pq.write_table(slice_table, slice_path, compression="zstd")

    item_ids, image, text, tags = load_catalog(args.scratch / "features/pixelrec50k")
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    context = pq.read_table(args.scratch / "contexts/pixelrec50k/item_context.parquet")
    fit_counts = context["fit_interactions"].to_numpy()
    tail = context["positive_support_tail"].to_numpy().astype(bool)
    first_seen = context["first_observed_timestamp"].to_numpy()
    boundary = max(row["query_timestamp"] for row in queries)
    eligible = first_seen <= boundary
    tag_one_hot = np.eye(int(tags.max()) + 1, dtype=np.float32)[tags]
    representation = np.concatenate([image, text, tag_one_hot], axis=1).astype(
        np.float32
    )
    representation /= np.maximum(
        np.linalg.norm(representation, axis=1, keepdims=True),
        np.finfo(np.float32).eps,
    )
    nonaccuracy = {
        model_id: _topk_metrics(
            rows,
            item_to_index,
            fit_counts,
            tail,
            eligible,
            representation,
        )
        for model_id, rows in model_rows.items()
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-F03-F05",
        "status": "PASS_DESCRIPTIVE",
        "role": "LOCKED_FINAL",
        "qualification_access_count": 1,
        "locked_final_access_count": 1,
        "input_sha256": {model: sha256_file(path) for model, path in paths.items()},
        "slice_table": {
            "path": slice_path.as_posix(),
            "sha256": sha256_file(slice_path),
            "rows": len(slice_table),
        },
        "non_accuracy": nonaccuracy,
        "personalization_disposition": "The frozen metric named deterministic SHA256-selected user pairs but did not define the pair-construction algorithm; no post-lock algorithm was invented.",
        "evidence_class": "p2_locked_preregistered_secondary_descriptive",
    }
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during locked secondary analysis")
    write_json_new(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
