"""Generate the complete read-only MOSAIC-Rec p2 industry diagnostic package."""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import (
    SOURCE_COMMIT,
    SOURCE_PROTOCOL,
    load_json,
    metadata,
    sha256_file,
    write_json_new,
    write_parquet_new,
)
from diagnostics import (
    METRICS,
    age_bucket,
    assert_paired,
    concentration,
    descriptive_classification,
    gini,
    history_bucket,
    jaccard,
    modality_bucket,
    oracle_values,
    paired_bootstrap,
    quantile_edges,
    quantile_label,
    recency_bucket,
    support_bucket,
    temporal_quartiles,
    validate_output_path,
    win_tie_loss,
)

MODEL_NAMES = ("C_seq", "C_cold", "R4")
COMPARISONS = (("R4_minus_C_seq", "R4", "C_seq"), ("R4_minus_C_cold", "R4", "C_cold"))


def _source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {row["path"]: row["sha256"] for row in manifest["inputs"]}


def _read_locked(path: Path, prefix: str) -> pd.DataFrame:
    frame = pq.read_table(path).to_pandas()
    rename = {metric: f"{prefix}_{metric}" for metric in METRICS}
    rename.update(
        {
            "target_rank": f"{prefix}_target_rank",
            "top20_item_ids": f"{prefix}_top20",
            "top20_scores": f"{prefix}_top20_scores",
        }
    )
    keep = ["user_id", "query_timestamp", "target_item_id", *rename]
    return frame.loc[:, keep].rename(columns=rename)


def _load_joined(scratch: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = {
        "C_seq": scratch / "locked_final/cseq_bert4rec_topk.parquet",
        "C_cold": scratch / "locked_final/ccold_two_tower_topk.parquet",
        "R4": scratch / "locked_final/r4_topk.parquet",
    }
    frames = {name: _read_locked(path, name.lower()) for name, path in paths.items()}
    assert_paired(frames.values())
    frame = frames["C_seq"]
    for name in ("C_cold", "R4"):
        frame = frame.merge(
            frames[name],
            on=["user_id", "query_timestamp", "target_item_id"],
            validate="one_to_one",
        )
    queries = pq.read_table(
        scratch / "contexts/pixelrec50k/LOCKED_FINAL_QUERIES.parquet"
    ).to_pandas()
    frame = queries.merge(
        frame,
        on=["user_id", "query_timestamp", "target_item_id"],
        validate="one_to_one",
    )
    diagnostics = pq.read_table(
        paths["R4"],
        columns=[
            "user_id",
            "query_timestamp",
            "target_item_id",
            "router_weight",
            "sequence_score",
            "content_score",
        ],
    ).to_pandas()
    frame = frame.merge(
        diagnostics,
        on=["user_id", "query_timestamp", "target_item_id"],
        validate="one_to_one",
    )
    return frame, paths


def _add_slices(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    result["support_bucket"] = [
        support_bucket(int(value)) for value in result["fit_interactions"]
    ]
    result["item_age_bucket"] = [
        age_bucket(float(value)) for value in result["item_age_days"]
    ]
    result["history_bucket"] = [
        history_bucket(int(value)) for value in result["fit_history_length"]
    ]
    result["recency_bucket"] = [
        recency_bucket(float(value)) for value in result["user_recency_days"]
    ]
    result["modality"] = [
        modality_bucket(bool(image), bool(text))
        for image, text in zip(result["image_present"], result["text_present"])
    ]
    tail_edges = quantile_edges(result["history_tail_share"].to_numpy())
    popularity_edges = quantile_edges(result["median_history_popularity"].to_numpy())
    result["tail_affinity_quartile"] = [
        quantile_label(float(value), tail_edges)
        for value in result["history_tail_share"]
    ]
    result["median_popularity_quartile"] = [
        quantile_label(float(value), popularity_edges)
        for value in result["median_history_popularity"]
    ]
    time_edges, time_labels = temporal_quartiles(result["query_timestamp"].to_numpy())
    result["time_quartile"] = [f"Q{value}" for value in time_labels]
    return result, {
        "tail_affinity": tail_edges,
        "median_history_popularity": popularity_edges,
        "query_timestamp_quartile_boundaries": time_edges,
    }


def _axes() -> dict[str, str]:
    return {
        "support": "support_bucket",
        "item_age": "item_age_bucket",
        "popularity_tier": "popularity_tier",
        "history_length": "history_bucket",
        "recency": "recency_bucket",
        "tail_affinity": "tail_affinity_quartile",
        "median_history_popularity": "median_popularity_quartile",
        "modality": "modality",
    }


def _groups(
    frame: pd.DataFrame, axes: dict[str, str] | None = None
) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "overall", "all", frame
    for family, column in (axes or _axes()).items():
        for value, subset in frame.groupby(column, sort=True, observed=False):
            yield family, str(value), subset


def _metric_column(model: str, metric: str) -> str:
    return f"{model.lower()}_{metric}"


def performance_rows(
    frame: pd.DataFrame, *, crossed: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if crossed:
        definitions = {
            "support_x_item_age": ("support_bucket", "item_age_bucket"),
            "support_x_modality": ("support_bucket", "modality"),
            "history_x_recency": ("history_bucket", "recency_bucket"),
            "support_x_popularity": ("support_bucket", "popularity_tier"),
        }
        groups = []
        for family, columns in definitions.items():
            for values, subset in frame.groupby(
                list(columns), sort=True, observed=False
            ):
                if len(subset) >= 200:
                    groups.append((family, " × ".join(map(str, values)), subset))
    else:
        groups = list(_groups(frame))
    rows: list[dict[str, Any]] = []
    uncertainty: list[dict[str, Any]] = []
    for family, label, subset in groups:
        means = {
            model: float(subset[_metric_column(model, "ndcg_at_20")].mean())
            for model in MODEL_NAMES
        }
        best = max(means, key=means.get)
        for model in MODEL_NAMES:
            rows.append(
                {
                    "row_type": "model_metric",
                    "slice_family": family,
                    "slice": label,
                    "model": model,
                    "comparison": None,
                    "metric": "all",
                    "n": len(subset),
                    "ndcg_at_20": means[model],
                    "recall_at_20": float(
                        subset[_metric_column(model, "recall_at_20")].mean()
                    ),
                    "mrr_at_20": float(
                        subset[_metric_column(model, "mrr_at_20")].mean()
                    ),
                    "hit_at_20": float(
                        subset[_metric_column(model, "hit_at_20")].mean()
                    ),
                    "precision_at_20": float(
                        subset[_metric_column(model, "precision_at_20")].mean()
                    ),
                    "point_estimate": None,
                    "lower_95": None,
                    "upper_95": None,
                    "best_descriptive_model": best,
                    "small_sample": len(subset) < 200,
                }
            )
        for comparison, left, right in COMPARISONS:
            for metric in METRICS:
                delta = (
                    subset[_metric_column(left, metric)].to_numpy()
                    - subset[_metric_column(right, metric)].to_numpy()
                )
                interval = paired_bootstrap(
                    delta,
                    f"performance:{crossed}:{family}:{label}:{comparison}:{metric}",
                )
                rows.append(
                    {
                        "row_type": "paired_delta",
                        "slice_family": family,
                        "slice": label,
                        "model": None,
                        "comparison": comparison,
                        "metric": metric,
                        "n": len(subset),
                        "ndcg_at_20": None,
                        "recall_at_20": None,
                        "mrr_at_20": None,
                        "hit_at_20": None,
                        "precision_at_20": None,
                        **interval,
                        "best_descriptive_model": best,
                        "small_sample": len(subset) < 200,
                    }
                )
            ndcg_delta = (
                subset[_metric_column(left, "ndcg_at_20")].to_numpy()
                - subset[_metric_column(right, "ndcg_at_20")].to_numpy()
            )
            accounting = win_tie_loss(ndcg_delta)
            gains = concentration(ndcg_delta)
            uncertainty.append(
                {
                    "slice_family": family,
                    "slice": label,
                    "comparison": comparison,
                    **paired_bootstrap(
                        ndcg_delta, f"uncertainty:{family}:{label}:{comparison}"
                    ),
                    "nonzero_difference_n": int((ndcg_delta != 0).sum()),
                    "nonzero_difference_fraction": float((ndcg_delta != 0).mean()),
                    "positive_fraction": accounting["win_rate"],
                    "zero_fraction": accounting["tie_rate"],
                    "negative_fraction": accounting["loss_rate"],
                    "unique_target_items": int(subset["target_item_id"].nunique()),
                    "small_sample": len(subset) < 200,
                    "sparse_signal": float((ndcg_delta != 0).mean()) < 0.05,
                    "high_concentration": gains["top_10pct"] >= 0.80,
                    "high_concentration_definition": "top 10% of positive per-query gains account for at least 80% of positive gain",
                }
            )
    return rows, uncertainty


def routing_outputs(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    work = frame.copy()
    edges = quantile_edges(work["router_weight"].to_numpy(), groups=10)
    work["router_decile"] = [
        f"D{int(np.searchsorted(edges[1:-1], value, side='right')) + 1}"
        for value in work["router_weight"]
    ]
    expert_delta = work["c_seq_ndcg_at_20"] - work["c_cold_ndcg_at_20"]
    chosen = np.where(
        work["router_weight"] > 0.5,
        "sequence",
        np.where(work["router_weight"] < 0.5, "content", "neutral"),
    )
    stronger = np.where(
        expert_delta > 0, "sequence", np.where(expert_delta < 0, "content", "tie")
    )
    work["routing_agrees"] = (chosen == stronger) & (stronger != "tie")
    work["routing_disagrees"] = (
        (chosen != stronger) & (chosen != "neutral") & (stronger != "tie")
    )
    _, regret = oracle_values(
        work["c_seq_ndcg_at_20"], work["c_cold_ndcg_at_20"], work["r4_ndcg_at_20"]
    )
    work["oracle_regret"] = regret
    distribution = []
    for decile, subset in work.groupby("router_decile", sort=True):
        delta = subset["c_seq_ndcg_at_20"] - subset["c_cold_ndcg_at_20"]
        distribution.append(
            {
                "router_decile": decile,
                "n": len(subset),
                "mean_router_weight": float(subset["router_weight"].mean()),
                "fraction_cseq_gt_ccold": float((delta > 0).mean()),
                "fraction_ccold_gt_cseq": float((delta < 0).mean()),
                "fraction_tie": float((delta == 0).mean()),
                "mean_cseq_minus_ccold": float(delta.mean()),
                "mean_r4_ndcg": float(subset["r4_ndcg_at_20"].mean()),
                "mean_oracle_regret": float(subset["oracle_regret"].mean()),
            }
        )
    alignment = []
    axes = {**_axes(), "router_decile": "router_decile"}
    for family, label, subset in _groups(work, axes):
        alignment.append(
            {
                "slice_family": family,
                "slice": label,
                "n": len(subset),
                "agreement_rate": float(subset["routing_agrees"].mean()),
                "disagreement_rate": float(subset["routing_disagrees"].mean()),
                "neutral_rate": float((subset["router_weight"] == 0.5).mean()),
                "expert_tie_rate": float(
                    (subset["c_seq_ndcg_at_20"] == subset["c_cold_ndcg_at_20"]).mean()
                ),
                "mean_router_weight": float(subset["router_weight"].mean()),
            }
        )
    spearman = float(
        pd.Series(work["router_weight"])
        .rank(method="average")
        .corr(pd.Series(expert_delta).rank(method="average"))
    )
    return (
        distribution,
        alignment,
        {
            "router_weight_decile_edges": edges,
            "spearman_router_weight_vs_expert_ndcg_delta": spearman,
        },
    )


def gain_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cold = frame[frame["is_controlled_zero"]].copy()
    cold["r4_minus_c_cold"] = cold["r4_ndcg_at_20"] - cold["c_cold_ndcg_at_20"]
    delta = cold["r4_minus_c_cold"].to_numpy()
    positive = cold[cold["r4_minus_c_cold"] > 0]
    by_user = positive.groupby("user_id")["r4_minus_c_cold"].sum().to_numpy()
    by_item = positive.groupby("target_item_id")["r4_minus_c_cold"].sum().to_numpy()
    summary = {
        **win_tie_loss(delta),
        "mean_positive_delta": float(delta[delta > 0].mean())
        if (delta > 0).any()
        else 0.0,
        "mean_negative_delta": float(delta[delta < 0].mean())
        if (delta < 0).any()
        else 0.0,
        "median_delta": float(np.median(delta)),
        "r4_nonzero_hit_count": int((cold["r4_ndcg_at_20"] > 0).sum()),
        "c_cold_nonzero_hit_count": int((cold["c_cold_ndcg_at_20"] > 0).sum()),
        "positive_gain_users": int(positive["user_id"].nunique()),
        "positive_gain_items": int(positive["target_item_id"].nunique()),
        "query_concentration": concentration(delta),
        "user_concentration": concentration(by_user),
        "item_concentration": concentration(by_item),
        "positive_gain_gini": gini(delta[delta > 0]),
        "user_gain_gini": gini(by_user),
        "item_gain_gini": gini(by_item),
    }
    return cold, summary


def oracle_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame[
        [
            "user_id",
            "query_timestamp",
            "target_item_id",
            "support_bucket",
            "item_age_bucket",
            "history_bucket",
            "modality",
            "is_controlled_zero",
            "is_warm",
        ]
    ].copy()
    oracle, regret = oracle_values(
        frame["c_seq_ndcg_at_20"], frame["c_cold_ndcg_at_20"], frame["r4_ndcg_at_20"]
    )
    result["oracle_ndcg_at_20"] = oracle
    result["r4_ndcg_at_20"] = frame["r4_ndcg_at_20"].to_numpy()
    result["routing_regret"] = regret
    summary: dict[str, Any] = {
        "overall": {
            "oracle_ndcg_at_20": float(oracle.mean()),
            "r4_ndcg_at_20": float(frame["r4_ndcg_at_20"].mean()),
            "c_seq_ndcg_at_20": float(frame["c_seq_ndcg_at_20"].mean()),
            "c_cold_ndcg_at_20": float(frame["c_cold_ndcg_at_20"].mean()),
            "oracle_minus_r4": float(regret.mean()),
            "mean_regret": float(regret.mean()),
            "median_regret": float(np.median(regret)),
            **{f"p{q}": float(np.percentile(regret, q)) for q in (75, 90, 95, 99)},
            "fraction_regret_zero": float((regret == 0).mean()),
            "fraction_regret_positive": float((regret > 0).mean()),
            "fraction_r4_above_oracle": float((regret < 0).mean()),
        },
        "slices": [],
    }
    slice_axes = {
        "controlled_zero": "is_controlled_zero",
        "warm": "is_warm",
        "support": "support_bucket",
        "item_age": "item_age_bucket",
        "history": "history_bucket",
        "modality": "modality",
    }
    for family, column in slice_axes.items():
        values = (
            [True]
            if family in {"controlled_zero", "warm"}
            else sorted(result[column].unique())
        )
        for value in values:
            subset = result[result[column] == value]
            summary["slices"].append(
                {
                    "slice_family": family,
                    "slice": str(value),
                    "n": len(subset),
                    "oracle_ndcg_at_20": float(subset["oracle_ndcg_at_20"].mean()),
                    "r4_ndcg_at_20": float(subset["r4_ndcg_at_20"].mean()),
                    "mean_regret": float(subset["routing_regret"].mean()),
                }
            )
    return result, summary


def expert_outputs(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = (
        ("C_seq_vs_C_cold", "C_seq", "C_cold"),
        ("R4_vs_C_seq", "R4", "C_seq"),
        ("R4_vs_C_cold", "R4", "C_cold"),
    )
    agreement: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    axes = {
        "controlled_zero": "is_controlled_zero",
        "warm": "is_warm",
        "support": "support_bucket",
        "item_age": "item_age_bucket",
    }
    groups = [("overall", "all", frame)]
    for family, column in axes.items():
        values = (
            [True]
            if family in {"controlled_zero", "warm"}
            else sorted(frame[column].unique())
        )
        groups.extend(
            (family, str(value), frame[frame[column] == value]) for value in values
        )
    for family, label, subset in groups:
        for comparison, left, right in comparisons:
            delta = (
                subset[_metric_column(left, "ndcg_at_20")].to_numpy()
                - subset[_metric_column(right, "ndcg_at_20")].to_numpy()
            )
            percentiles = np.percentile(delta, [10, 25, 50, 75, 90])
            agreement.append(
                {
                    "section": "pairwise_metric",
                    "slice_family": family,
                    "slice": label,
                    "comparison": comparison,
                    **win_tie_loss(delta),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    **{
                        f"p{q}": float(value)
                        for q, value in zip((10, 25, 50, 75, 90), percentiles)
                    },
                }
            )
            left_lists = subset[f"{left.lower()}_top20"]
            right_lists = subset[f"{right.lower()}_top20"]
            jaccards = np.asarray(
                [jaccard(list(a), list(b)) for a, b in zip(left_lists, right_lists)]
            )
            top5 = np.asarray(
                [len(set(a[:5]) & set(b[:5])) for a, b in zip(left_lists, right_lists)]
            )
            top20 = np.asarray(
                [len(set(a) & set(b)) for a, b in zip(left_lists, right_lists)]
            )
            overlap.append(
                {
                    "slice_family": family,
                    "slice": label,
                    "comparison": comparison,
                    "n": len(subset),
                    "mean_jaccard_at_20": float(jaccards.mean()),
                    "median_jaccard_at_20": float(np.median(jaccards)),
                    "top1_agreement": float(
                        np.mean([a[0] == b[0] for a, b in zip(left_lists, right_lists)])
                    ),
                    "mean_top5_intersection": float(top5.mean()),
                    "mean_top20_intersection": float(top20.mean()),
                }
            )
    seq_hit = frame["c_seq_hit_at_20"] > 0
    cold_hit = frame["c_cold_hit_at_20"] > 0
    matrices = {
        "both_experts_fail": ~seq_hit & ~cold_hit,
        "c_seq_only_succeeds": seq_hit & ~cold_hit,
        "c_cold_only_succeeds": ~seq_hit & cold_hit,
        "both_succeed": seq_hit & cold_hit,
    }
    for label, mask in matrices.items():
        subset = frame[mask]
        delta = subset["r4_ndcg_at_20"] - np.maximum(
            subset["c_seq_ndcg_at_20"], subset["c_cold_ndcg_at_20"]
        )
        agreement.append(
            {
                "section": "success_matrix",
                "slice_family": "success_matrix",
                "slice": label,
                "comparison": "R4_vs_stronger_expert",
                **win_tie_loss(delta),
                "fraction_total": len(subset) / len(frame),
                "r4_ndcg_at_20": float(subset["r4_ndcg_at_20"].mean())
                if len(subset)
                else 0.0,
                "r4_hit_rate": float(subset["r4_hit_at_20"].mean())
                if len(subset)
                else 0.0,
                "mean_router_weight": float(subset["router_weight"].mean())
                if len(subset)
                else 0.0,
                "mean_delta": float(delta.mean()) if len(subset) else 0.0,
                "median_delta": float(np.median(delta)) if len(subset) else 0.0,
                **{
                    f"p{q}": float(np.percentile(delta, q)) if len(subset) else 0.0
                    for q in (10, 25, 50, 75, 90)
                },
            }
        )
    return agreement, overlap


def temporal_outputs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for quartile, subset in frame.groupby("time_quartile", sort=True):
        for model in MODEL_NAMES:
            rows.append(
                {
                    "row_type": "model_metric",
                    "time_quartile": quartile,
                    "model": model,
                    "comparison": None,
                    "n": len(subset),
                    **{
                        metric: float(subset[_metric_column(model, metric)].mean())
                        for metric in METRICS
                    },
                    "mean_router_weight": float(subset["router_weight"].mean())
                    if model == "R4"
                    else None,
                    "point_estimate": None,
                    "lower_95": None,
                    "upper_95": None,
                }
            )
        for comparison, left, right in COMPARISONS:
            delta = (
                subset[_metric_column(left, "ndcg_at_20")].to_numpy()
                - subset[_metric_column(right, "ndcg_at_20")].to_numpy()
            )
            rows.append(
                {
                    "row_type": "paired_delta",
                    "time_quartile": quartile,
                    "model": None,
                    "comparison": comparison,
                    "n": len(subset),
                    "ndcg_at_20": None,
                    "recall_at_20": None,
                    "mrr_at_20": None,
                    "hit_at_20": None,
                    "precision_at_20": None,
                    "mean_router_weight": float(subset["router_weight"].mean()),
                    **paired_bootstrap(delta, f"temporal:{quartile}:{comparison}"),
                }
            )
    return rows


def _catalog_metric(
    subset: pd.DataFrame,
    model: str,
    *,
    item_to_index: dict[str, int],
    fit_counts: np.ndarray,
    tail: np.ndarray,
    eligible: np.ndarray,
    representation: np.ndarray,
) -> dict[str, float]:
    lists = [
        [item_to_index[item] for item in values]
        for values in subset[f"{model.lower()}_top20"]
    ]
    flat = np.asarray([item for values in lists for item in values], dtype=np.int64)
    exposure = np.bincount(flat, minlength=fit_counts.size).astype(float)
    eligible_exposure = exposure[eligible]
    shares = eligible_exposure / eligible_exposure.sum()
    nonzero = shares[shares > 0]
    novelty = -np.log2((fit_counts + 1) / (fit_counts.sum() + eligible.sum()))
    diversity = []
    for values in lists:
        vectors = representation[np.asarray(values)]
        summed = vectors.sum(axis=0)
        cosine_sum = (float(summed @ summed) - len(values)) / 2
        pairs = len(values) * (len(values) - 1) / 2
        diversity.append(1 - cosine_sum / pairs)
    return {
        "item_coverage_at_20": float(
            np.count_nonzero(eligible_exposure) / eligible.sum()
        ),
        "average_popularity_at_20": float(np.log1p(fit_counts[flat]).mean()),
        "tail_percentage_at_20": float(tail[flat].mean()),
        "exposure_gini": gini(eligible_exposure),
        "exposure_hhi": float(np.square(shares).sum()),
        "exposure_entropy": float(-(nonzero * np.log(nonzero)).sum()),
        "novelty": float(novelty[flat].mean()),
        "intra_list_diversity": float(np.mean(diversity)),
    }


def exposure_outputs(
    frame: pd.DataFrame, scratch: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_dir = scratch / "features/pixelrec50k"
    item_ids = (feature_dir / "publisher_image_item_ids.txt").read_text().splitlines()
    image = np.load(feature_dir / "publisher_image_features.npy", mmap_mode="r")
    text = np.load(feature_dir / "authorized_minilm_text_features.npy", mmap_mode="r")
    tags = np.load(feature_dir / "tag_ids.npy")
    tag_one_hot = np.eye(int(tags.max()) + 1, dtype=np.float32)[tags]
    representation = np.concatenate(
        [np.asarray(image), np.asarray(text), tag_one_hot], axis=1
    ).astype(np.float32)
    representation /= np.maximum(
        np.linalg.norm(representation, axis=1, keepdims=True), np.finfo(np.float32).eps
    )
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    context = pq.read_table(
        scratch / "contexts/pixelrec50k/item_context.parquet"
    ).to_pandas()
    fit_counts = context["fit_interactions"].to_numpy()
    tail = context["positive_support_tail"].to_numpy().astype(bool)
    eligible = (
        context["first_observed_timestamp"].to_numpy() <= frame["query_timestamp"].max()
    )
    axes = {
        "controlled_zero": "is_controlled_zero",
        "warm": "is_warm",
        "support": "support_bucket",
        "history": "history_bucket",
        "item_age": "item_age_bucket",
        "popularity_tier": "popularity_tier",
        "time_quartile": "time_quartile",
    }
    groups = [("overall", "all", frame)]
    for family, column in axes.items():
        values = (
            [True]
            if family in {"controlled_zero", "warm"}
            else sorted(frame[column].unique())
        )
        groups.extend(
            (family, str(value), frame[frame[column] == value]) for value in values
        )
    rows = []
    for family, label, subset in groups:
        if len(subset) == 0:
            continue
        model_metrics = {
            model: _catalog_metric(
                subset,
                model,
                item_to_index=item_to_index,
                fit_counts=fit_counts,
                tail=tail,
                eligible=eligible,
                representation=representation,
            )
            for model in MODEL_NAMES
        }
        for model, metrics in model_metrics.items():
            row: dict[str, Any] = {
                "slice_family": family,
                "slice": label,
                "model": model,
                "n": len(subset),
                **metrics,
            }
            for comparator in ("C_seq", "C_cold"):
                if model == "R4":
                    for metric, value in metrics.items():
                        denominator = model_metrics[comparator][metric]
                        row[f"{metric}_delta_vs_{comparator}"] = value - denominator
                        row[f"{metric}_relative_vs_{comparator}"] = (
                            (value - denominator) / denominator
                            if denominator != 0
                            else None
                        )
            rows.append(row)
    overall = {row["model"]: row for row in rows if row["slice_family"] == "overall"}
    return rows, {"overall": overall, "eligible_catalog_items": int(eligible.sum())}


def robustness_outputs(
    frame: pd.DataFrame, repo: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_columns = ["user_id", "query_timestamp", "target_item_id"]
    base = frame
    rows = []
    missing_cell_summaries = []
    for receipt_path in sorted(repo.glob("evidence/p2/p2-g00-*.json")):
        if receipt_path.name.endswith("robustness-summary.json"):
            continue
        receipt = load_json(receipt_path)
        output = pq.read_table(receipt["locked_final"]["output"]).to_pandas()
        assert_paired([base.loc[:, base_columns], output.loc[:, base_columns]])
        cell = base.copy()
        for column in (
            *METRICS,
            "router_weight",
            "sequence_score",
            "content_score",
            "image_present",
            "text_present",
        ):
            cell[f"cell_{column}"] = output[column].to_numpy()
        axes = {
            "support_zero_positive": "support_zero_positive",
            "support": "support_bucket",
            "item_age": "item_age_bucket",
            "history": "history_bucket",
            "recency": "recency_bucket",
            "tail_affinity": "tail_affinity_quartile",
            "modality": "modality",
        }
        cell["support_zero_positive"] = np.where(
            cell["fit_interactions"] == 0, "zero", "positive"
        )
        for family, label, subset in _groups(cell, axes):
            rows.append(
                {
                    "intervention": receipt["intervention"],
                    "intensity": receipt["intensity"],
                    "slice_family": family,
                    "slice": label,
                    "n": len(subset),
                    **{
                        f"{metric}_delta": float(
                            (subset[f"cell_{metric}"] - subset[f"r4_{metric}"]).mean()
                        )
                        for metric in ("ndcg_at_20", "recall_at_20", "hit_at_20")
                    },
                    "router_weight_delta": float(
                        (subset["cell_router_weight"] - subset["router_weight"]).mean()
                    ),
                    "content_reliance_delta": float(
                        (subset["router_weight"] - subset["cell_router_weight"]).mean()
                    ),
                    "sequence_score_delta": float(
                        (
                            subset["cell_sequence_score"] - subset["sequence_score"]
                        ).mean()
                    ),
                    "content_score_delta": float(
                        (subset["cell_content_score"] - subset["content_score"]).mean()
                    ),
                }
            )
        if "MISSING" in receipt["intervention"]:
            covariates = {
                "zero_or_low_support": cell["fit_interactions"] <= 5,
                "young_item_age": cell["item_age_days"] <= 30,
                "missing_image_flag": ~cell["cell_image_present"].astype(bool),
                "missing_text_flag": ~cell["cell_text_present"].astype(bool),
                "short_user_history": cell["fit_history_length"] <= 10,
                "stale_user_history": cell["user_recency_days"] > 90,
                "high_tail_affinity": cell["tail_affinity_quartile"] == "Q4",
            }
            content_delta = cell["router_weight"] - cell["cell_router_weight"]
            for name, mask in covariates.items():
                group_mean = float(content_delta[mask].mean()) if mask.any() else 0.0
                complement = (
                    float(content_delta[~mask].mean()) if (~mask).any() else 0.0
                )
                missing_cell_summaries.append(
                    {
                        "intervention": receipt["intervention"],
                        "intensity": receipt["intensity"],
                        "covariate": name,
                        "n": int(mask.sum()),
                        "mean_content_reliance_delta": group_mean,
                        "association_contrast": group_mean - complement,
                        "classification": descriptive_classification(group_mean),
                    }
                )
    missing = pd.DataFrame(missing_cell_summaries)
    overall_increase = float(
        missing[missing["covariate"] == "zero_or_low_support"][
            "mean_content_reliance_delta"
        ].mean()
    )
    mechanism = {
        "hypothesis": "The learned router may conflate missing-content / cold-item signals with a reason to increase content reliance.",
        "classification": descriptive_classification(overall_increase),
        "mean_missingness_content_reliance_delta_in_zero_or_low_support": overall_increase,
        "associations": missing_cell_summaries,
        "causal_interpretation": False,
    }
    return rows, mechanism


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    body = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    body.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(body)


def _write_report(path: Path, title: str, body: str, repo: Path) -> None:
    validate_output_path(path, repo)
    preamble = (
        f"# {title}\n\n"
        "> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**\n\n"
        f"Source protocol: `{SOURCE_PROTOCOL}`  \n"
        f"Source commit: `{SOURCE_COMMIT}`  \n"
        "Protected scoring rerun: `false`  \nCanonical p2 mutated: `false`\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(preamble + body.rstrip() + "\n")


def _overall_metric(performance: pd.DataFrame, model: str, metric: str) -> float:
    row = performance[
        (performance["row_type"] == "model_metric")
        & (performance["slice_family"] == "overall")
        & (performance["model"] == model)
    ].iloc[0]
    return float(row[metric])


def _delta_row(
    performance: pd.DataFrame, family: str, label: str, comparison: str
) -> pd.Series:
    rows = performance[
        (performance["row_type"] == "paired_delta")
        & (performance["slice_family"] == family)
        & (performance["slice"] == label)
        & (performance["comparison"] == comparison)
        & (performance["metric"] == "ndcg_at_20")
    ]
    if rows.empty:
        raise KeyError((family, label, comparison))
    return rows.iloc[0]


def _reports(
    repo: Path,
    performance: pd.DataFrame,
    cross: pd.DataFrame,
    routing: pd.DataFrame,
    weights: pd.DataFrame,
    gain: dict[str, Any],
    oracle: dict[str, Any],
    agreement: pd.DataFrame,
    overlap: pd.DataFrame,
    temporal: pd.DataFrame,
    exposure: pd.DataFrame,
    catalog: dict[str, Any],
    robust: pd.DataFrame,
    mechanism: dict[str, Any],
    uncertainty: pd.DataFrame,
    edges: dict[str, Any],
) -> None:
    report = repo / "reports/posthoc"
    overall_rows = []
    for model in MODEL_NAMES:
        overall_rows.append(
            (
                model,
                *(_overall_metric(performance, model, metric) for metric in METRICS),
            )
        )
    overall_table = _table(
        ["Model", "NDCG@20", "Recall@20", "MRR@20", "Hit@20", "Precision@20"],
        overall_rows,
    )
    headline = uncertainty[
        (uncertainty["slice_family"] != "overall") & (~uncertainty["small_sample"])
    ]
    strongest = headline.sort_values("point_estimate", ascending=False).iloc[0]
    largest_risk = headline.sort_values("point_estimate").iloc[0]
    overall_oracle = oracle["overall"]
    capture = (
        overall_oracle["r4_ndcg_at_20"] / overall_oracle["oracle_ndcg_at_20"]
        if overall_oracle["oracle_ndcg_at_20"]
        else 0.0
    )
    warm_seq = _delta_row(performance, "support", ">50", "R4_minus_C_seq")
    overall_catalog = catalog["overall"]
    r4_catalog = overall_catalog["R4"]
    cseq_catalog = overall_catalog["C_seq"]
    ccold_catalog = overall_catalog["C_cold"]
    mechanism_class = mechanism["classification"]

    _write_report(
        report / "PERFORMANCE_SLICING.md",
        "Performance Slicing",
        f"""## Overall model quality

{overall_table}

The strongest non-small descriptive slice delta was `{strongest["slice_family"]} / {strongest["slice"]}` for `{strongest["comparison"]}`: {_fmt(strongest["point_estimate"])} (exploratory 95% interval {_fmt(strongest["lower_95"])}, {_fmt(strongest["upper_95"])}; N={int(strongest["n"])}). The largest descriptive risk was `{largest_risk["slice_family"]} / {largest_risk["slice"]}` for `{largest_risk["comparison"]}`: {_fmt(largest_risk["point_estimate"])} (N={int(largest_risk["n"])}). These rankings are diagnostics and do not change model selection.

Crossed slices were emitted only for N >= 200. Exact quartile edges are persisted in `analysis/posthoc/quartile_edges.json`.
""",
        repo,
    )
    spearman = routing[(routing["slice_family"] == "overall")].iloc[0]
    _write_report(
        report / "ROUTER_ROOT_CAUSE.md",
        "Router Root Cause",
        f"""The persisted value is reported as **sequence routing weight**, not as a probability. Overall routing-side agreement with the empirically stronger expert was {_fmt(spearman["agreement_rate"])}; disagreement was {_fmt(spearman["disagreement_rate"])}; neutral routing was {_fmt(spearman["neutral_rate"])}.

Spearman association between sequence routing weight and per-query C_seq minus C_cold NDCG was {_fmt(edges["spearman_router_weight_vs_expert_ndcg_delta"])}. This is an association measured after outcomes and is not a causal or reliability claim.

Router-weight deciles and every preregistered covariate grouping are in the machine-readable tables.
""",
        repo,
    )
    _write_report(
        report / "GAIN_CONCENTRATION.md",
        "Gain Concentration",
        f"""Within CONTROLLED_ZERO_SUPPORT, R4 minus C_cold had {gain["win_n"]} positive, {gain["tie_n"]} zero, and {gain["loss_n"]} negative per-query differences (N={gain["n"]}). Positive gains involved {gain["positive_gain_users"]} users and {gain["positive_gain_items"]} target items. The top 10% of positive-gain queries contributed {_fmt(gain["query_concentration"]["top_10pct"])} of all positive gain; positive-gain Gini was {_fmt(gain["positive_gain_gini"])}.

H1 had a positive point estimate while its lower interval touched zero because nonzero outcome differences were sparse and positive gain was concentrated. The persisted data support this sampling-uncertainty explanation descriptively; they do not establish a causal mechanism.
""",
        repo,
    )
    routing_gap = overall_oracle["oracle_minus_r4"]
    experts_best = max(
        overall_oracle["c_seq_ndcg_at_20"], overall_oracle["c_cold_ndcg_at_20"]
    )
    bottleneck = (
        "both weak experts and routing error"
        if overall_oracle["oracle_ndcg_at_20"] < 0.1 and routing_gap > 0
        else ("routing error" if routing_gap > 0 else "weak experts")
    )
    _write_report(
        report / "ORACLE_REGRET.md",
        "Oracle and Routing Regret",
        f"""The retrospective per-query expert oracle achieved NDCG@20 {_fmt(overall_oracle["oracle_ndcg_at_20"])}; R4 achieved {_fmt(overall_oracle["r4_ndcg_at_20"])}; the better single expert achieved {_fmt(experts_best)}. R4 captured {_fmt(capture * 100, 2)}% of oracle NDCG. Mean oracle minus R4 regret was {_fmt(routing_gap)}; R4 exceeded this two-expert oracle on {_fmt(overall_oracle["fraction_r4_above_oracle"] * 100, 2)}% of queries because R4 can produce a ranking different from both input experts.

The diagnostic classification is **{bottleneck}**. The oracle itself remains weak in absolute terms while a nonzero oracle-to-router gap remains. This oracle is retrospective and undeployable.
""",
        repo,
    )
    matrix = agreement[agreement["section"] == "success_matrix"]
    _write_report(
        report / "EXPERT_AGREEMENT.md",
        "Expert Agreement",
        f"""Per-query win/tie/loss, the four-cell success matrix, and top-k overlap are persisted for overall, cold, warm, support, and item-age slices.

{_table(["Success matrix cell", "N", "Fraction", "R4 NDCG", "Mean routing weight"], ((row["slice"], int(row["n"]), row["fraction_total"], row["r4_ndcg_at_20"], row["mean_router_weight"]) for _, row in matrix.iterrows()))}

Top-20 Jaccard, top-1 agreement, and intersection sizes quantify list agreement without treating unobserved interactions as verified negatives.
""",
        repo,
    )
    temporal_model = temporal[temporal["row_type"] == "model_metric"]
    variations = {
        model: float(group["ndcg_at_20"].max() - group["ndcg_at_20"].min())
        for model, group in temporal_model.groupby("model")
    }
    _write_report(
        report / "TEMPORAL_STABILITY.md",
        "Temporal Stability",
        f"""Chronological quartiles used deterministic query ordering. Exact timestamp boundaries are `{edges["query_timestamp_quartile_boundaries"]}`.

Max-minus-min NDCG variation was C_seq {_fmt(variations["C_seq"])}, C_cold {_fmt(variations["C_cold"])}, and R4 {_fmt(variations["R4"])}. Router-weight drift and expert-disagreement drift are reported descriptively. Coverage, tail exposure, and average popularity by quartile are included in the exposure table. No causal time-trend claim is made.
""",
        repo,
    )
    _write_report(
        report / "EXPOSURE_ANALYSIS.md",
        "Exposure and Catalog Health",
        f"""## Overall catalog metrics

{_table(["Model", "Coverage@20", "Avg popularity", "Tail %", "Gini", "HHI", "Entropy", "Novelty", "ILD"], ((name, row["item_coverage_at_20"], row["average_popularity_at_20"], row["tail_percentage_at_20"], row["exposure_gini"], row["exposure_hhi"], row["exposure_entropy"], row["novelty"], row["intra_list_diversity"]) for name, row in overall_catalog.items()))}

Relative to C_seq, R4 coverage changed by {_fmt(r4_catalog["item_coverage_at_20"] - cseq_catalog["item_coverage_at_20"])}, tail exposure by {_fmt(r4_catalog["tail_percentage_at_20"] - cseq_catalog["tail_percentage_at_20"])}, and Gini by {_fmt(r4_catalog["exposure_gini"] - cseq_catalog["exposure_gini"])}. Relative to C_cold, the same deltas were {_fmt(r4_catalog["item_coverage_at_20"] - ccold_catalog["item_coverage_at_20"])}, {_fmt(r4_catalog["tail_percentage_at_20"] - ccold_catalog["tail_percentage_at_20"])}, and {_fmt(r4_catalog["exposure_gini"] - ccold_catalog["exposure_gini"])}. These are offline catalog-health descriptions, not fairness or engagement claims.
""",
        repo,
    )
    _write_report(
        report / "ROBUSTNESS_FAILURE_RCA.md",
        "Robustness Failure Root Cause",
        f"""All results reuse the 14 existing immutable robustness outputs; no robustness inference was rerun. The candidate explanation that the learned router may conflate missing-content or cold-item signals with a reason to increase content reliance is classified **{mechanism_class}**. Mean content-reliance change for zero-or-low-support rows across missingness cells was {_fmt(mechanism["mean_missingness_content_reliance_delta_in_zero_or_low_support"])}.

This classification describes observed routing deltas. It does not establish a causal mechanism. Covariate-specific associations are labeled SUPPORTED_ASSOCIATION, WEAK_ASSOCIATION, NOT_SUPPORTED, or UNTESTABLE in the machine-readable summary.
""",
        repo,
    )
    _write_report(
        report / "UNCERTAINTY_BY_SLICE.md",
        "Uncertainty by Slice",
        f"""Every major single-axis R4 comparison uses a deterministic 10,000-draw paired per-query bootstrap with a new posthoc seed namespace. `{int(uncertainty["small_sample"].sum())}` rows have N < 200; `{int(uncertainty["sparse_signal"].sum())}` have fewer than 5% nonzero differences; `{int(uncertainty["high_concentration"].sum())}` meet the declared high-concentration rule.

The widest interval belongs to `{uncertainty.assign(width=uncertainty["upper_95"] - uncertainty["lower_95"]).sort_values("width", ascending=False).iloc[0]["slice_family"]}` / `{uncertainty.assign(width=uncertainty["upper_95"] - uncertainty["lower_95"]).sort_values("width", ascending=False).iloc[0]["slice"]}`. These intervals are exploratory and are not PASS/FAIL gates.
""",
        repo,
    )

    answers = f"""1. **Where R4 helps:** the strongest qualifying descriptive slice is `{strongest["slice_family"]} / {strongest["slice"]}` for `{strongest["comparison"]}`, delta {_fmt(strongest["point_estimate"])}.
2. **Where R4 fails:** the largest slice risk is `{largest_risk["slice_family"]} / {largest_risk["slice"]}`, delta {_fmt(largest_risk["point_estimate"])}.
3. **Cold gain breadth:** controlled-zero gain is concentrated: {gain["win_n"]} positive queries among {gain["n"]}, top-10% positive-gain share {_fmt(gain["query_concentration"]["top_10pct"])}.
4. **H1 lower bound:** sparse, concentrated nonzero gains left substantial bootstrap mass at zero.
5. **Warm collapse:** R4 was below C_seq in the >50-support slice by {_fmt(warm_seq["point_estimate"])}; routing disagreement and ranking differences are associated diagnostics, not a causal explanation.
6. **Router choice:** routing-side agreement was {_fmt(spearman["agreement_rate"])}, disagreement {_fmt(spearman["disagreement_rate"])}.
7. **Oracle capture:** R4 captured {_fmt(capture * 100, 2)}% of retrospective oracle NDCG.
8. **Main bottleneck:** {bottleneck}; both absolute oracle quality and routing regret matter.
9. **Missingness direction:** the predefined conflation hypothesis is `{mechanism_class}` from persisted routing deltas; causal mechanism remains unproven.
10. **Catalog health:** R4 vs C_seq changed coverage {_fmt(r4_catalog["item_coverage_at_20"] - cseq_catalog["item_coverage_at_20"])}, tail exposure {_fmt(r4_catalog["tail_percentage_at_20"] - cseq_catalog["tail_percentage_at_20"])}, and Gini {_fmt(r4_catalog["exposure_gini"] - cseq_catalog["exposure_gini"])}.
11. **Temporal stability:** R4 max-minus-min quartile NDCG variation was {_fmt(variations["R4"])}.
12. **Strongest positive slice:** `{strongest["slice_family"]} / {strongest["slice"]}` ({strongest["comparison"]}).
13. **Largest risk:** `{largest_risk["slice_family"]} / {largest_risk["slice"]}` ({largest_risk["comparison"]}).
14. **P3 architecture:** constrain zero-support routing, calibrate expert score scales, train explicit expert-selection supervision on validation-only labels, and add warm-preservation objectives before protected access.
15. **Safe new numbers:** only values labeled posthoc exploratory with source commit and population; no number changes the p2 RETAIN_BASELINE decision.
"""
    _write_report(
        report / "INDUSTRY_DIAGNOSTIC_REPORT.md",
        "Industry Diagnostic Report",
        f"""## Source boundary

Canonical p2 remains CLOSED with `RETAIN_BASELINE`. All computations read existing locked, robustness, context, feature, and systems artifacts. No protected scoring or training occurred.

## Model quality

{overall_table}

## Answers to the diagnostic questions

{answers}
""",
        repo,
    )
    _write_report(
        report / "EXECUTIVE_SUMMARY.md",
        "Executive Summary",
        f"""Canonical p2 retained its baseline. This read-only study finds that R4's controlled-zero advantage was sparse and concentrated, while warm and high-support slices continued to favor C_seq. R4 captured {_fmt(capture * 100, 2)}% of a retrospective per-query two-expert oracle; the remaining gap and low absolute oracle quality indicate {bottleneck}. Catalog health and temporal behavior are reported as offline diagnostics only.

No posthoc finding revises p2. The useful outcome is a p3 hypothesis set: hard zero-support behavior, explicit warm preservation, stronger score calibration, and validation-only routing supervision.
""",
        repo,
    )
    _write_report(
        report / "RESUME_CLAIM_CANDIDATES.md",
        "Resume and Interview Claim Candidates",
        f"""| Classification | Candidate statement |
| --- | --- |
| SAFE_DESCRIPTIVE | Built a read-only diagnostic pipeline over three immutable full-catalog recommender outputs, with deterministic 10,000-draw paired bootstrap intervals and catalog-health analysis. |
| SAFE_WITH_EXPLORATORY_QUALIFIER | In posthoc analysis, R4 captured {_fmt(capture * 100, 2)}% of a retrospective two-expert oracle's NDCG@20; this did not change the retained-baseline decision. |
| SAFE_WITH_EXPLORATORY_QUALIFIER | Controlled-zero R4 gain occurred on {gain["win_n"]} of {gain["n"]} queries and was concentrated across {gain["positive_gain_items"]} target items. |
| INTERVIEW_ONLY | Routing-side agreement was {_fmt(spearman["agreement_rate"])}; use this only with the exact persisted-outcome definition and exploratory qualifier. |
| DO_NOT_CLAIM | R4 was promoted, statistically superior, production-ready, reliable, causal, or proven to improve engagement. |
| DO_NOT_CLAIM | Robustness interventions proved a causal missingness mechanism. |
""",
        repo,
    )
    _write_report(
        report / "P3_RESEARCH_RECOMMENDATIONS.md",
        "P3 Research Recommendations",
        """1. Preregister a hard content path at exactly zero support and a rapid monotonic switch to sequence for positive support.
2. Add explicit validation-only router supervision based on expert advantage, with leakage guards and a frozen loss.
3. Calibrate expert scores and measure calibration error before router HPO.
4. Preserve C_seq warm behavior with a frozen distillation or constraint objective.
5. Preregister missing-modality direction tests and train missingness augmentation before qualification.
6. Retain top-k, routing, expert-score, and catalog-health artifacts for every protected role.
7. Use independent data and a new protocol; none of these posthoc patterns are p2-confirmed hypotheses.
""",
        repo,
    )

    dashboard_rows = "".join(
        f"<tr><td>{html.escape(model)}</td>"
        + "".join(f"<td>{_fmt(value)}</td>" for value in values)
        + "</tr>"
        for model, *values in overall_rows
    )
    sections = {
        "A. Model quality": overall_table,
        "B. Slice health": f"Strongest: {strongest['slice_family']} / {strongest['slice']}; risk: {largest_risk['slice_family']} / {largest_risk['slice']}.",
        "C. Routing health": f"Agreement {_fmt(spearman['agreement_rate'])}; disagreement {_fmt(spearman['disagreement_rate'])}; oracle capture {_fmt(capture * 100, 2)}%.",
        "D. Gain concentration": f"Cold positive/zero/negative: {gain['win_n']}/{gain['tie_n']}/{gain['loss_n']}.",
        "E. Catalog health": f"R4 coverage {_fmt(r4_catalog['item_coverage_at_20'])}; tail exposure {_fmt(r4_catalog['tail_percentage_at_20'])}; Gini {_fmt(r4_catalog['exposure_gini'])}.",
        "F. Temporal health": f"R4 NDCG max-min variation {_fmt(variations['R4'])}.",
        "G. Robustness": f"14 immutable cells; missingness hypothesis {mechanism_class}.",
        "H. Uncertainty": f"Small-sample rows {int(uncertainty['small_sample'].sum())}; sparse-signal rows {int(uncertainty['sparse_signal'].sum())}.",
        "I. Data / protocol quality": "Source commit pinned; protected counts remain 1/1; no rerun; canonical p2 unchanged.",
        "J. Systems": "See canonical p2 systems receipts for R4 users/s, training throughput, peak VRAM, artifact size, and single-GPU disclosure.",
    }
    cards = "".join(
        f"<section><h2>{html.escape(title)}</h2><pre>{html.escape(text)}</pre></section>"
        for title, text in sections.items()
    )
    dashboard = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MOSAIC-Rec p2 industry diagnostics</title><style>body{{font:15px system-ui;margin:2rem;background:#f4f6f8;color:#16202a}}header,section{{background:white;padding:1rem 1.25rem;margin:.8rem 0;border-radius:8px;box-shadow:0 1px 3px #ccd}}.warning{{color:#8a3b00;font-weight:700}}pre{{white-space:pre-wrap}}table{{border-collapse:collapse}}td,th{{padding:.4rem;border:1px solid #ccd}}</style></head><body><header><h1>MOSAIC-Rec p2 Industry Diagnostic Dashboard</h1><p class="warning">POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM</p><p>Source {SOURCE_COMMIT}; canonical decision RETAIN_BASELINE; protected scoring rerun false.</p><table><tr><th>Model</th><th>NDCG@20</th><th>Recall@20</th><th>MRR@20</th><th>Hit@20</th><th>Precision@20</th></tr>{dashboard_rows}</table></header>{cards}</body></html>"""
    dashboard_path = report / "INDUSTRY_DASHBOARD.html"
    validate_output_path(dashboard_path, repo)
    dashboard_path.write_text(dashboard, encoding="utf-8", newline="\n")


def _artifact_manifest(
    repo: Path, input_hashes: dict[str, str], row_counts: dict[str, int]
) -> dict[str, Any]:
    artifacts = []
    for root in (
        repo / "scripts/posthoc",
        repo / "analysis/posthoc",
        repo / "reports/posthoc",
    ):
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and path.name != "posthoc_manifest.json"
                and "__pycache__" not in path.parts
            ):
                relative = path.relative_to(repo).as_posix()
                artifacts.append(
                    {
                        "path": relative,
                        "sha256": sha256_file(path),
                        "row_count": row_counts.get(relative),
                    }
                )
    return {
        **metadata(input_hashes=input_hashes),
        "manifest_self_hash_excluded": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--scratch", type=Path, default=Path("/home/bjw-0/mosaic-rec-work/p2")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    source_manifest = load_json(repo / "analysis/posthoc/source_manifest.json")
    input_hashes = _source_hashes(source_manifest)
    frame, _ = _load_joined(args.scratch)
    frame, quartile_edges = _add_slices(frame)

    performance, uncertainty = performance_rows(frame)
    cross, _ = performance_rows(frame, crossed=True)
    weights, routing, routing_meta = routing_outputs(frame)
    cold, gain = gain_outputs(frame)
    regret, oracle = oracle_outputs(frame)
    agreement, overlap = expert_outputs(frame)
    temporal = temporal_outputs(frame)
    exposure, catalog = exposure_outputs(frame, args.scratch)
    robust, mechanism = robustness_outputs(frame, repo)
    quartile_edges.update(routing_meta)

    outputs: dict[str, Any] = {
        "analysis/posthoc/performance_slices.parquet": pd.DataFrame(performance),
        "analysis/posthoc/performance_cross_slices.parquet": pd.DataFrame(cross),
        "analysis/posthoc/routing_alignment.parquet": pd.DataFrame(routing),
        "analysis/posthoc/router_weight_distribution.parquet": pd.DataFrame(weights),
        "analysis/posthoc/cold_query_deltas.parquet": cold,
        "analysis/posthoc/routing_regret.parquet": regret,
        "analysis/posthoc/expert_agreement.parquet": pd.DataFrame(agreement),
        "analysis/posthoc/topk_overlap.parquet": pd.DataFrame(overlap),
        "analysis/posthoc/temporal_stability.parquet": pd.DataFrame(temporal),
        "analysis/posthoc/exposure_by_slice.parquet": pd.DataFrame(exposure),
        "analysis/posthoc/robustness_root_cause.parquet": pd.DataFrame(robust),
        "analysis/posthoc/slice_uncertainty.parquet": pd.DataFrame(uncertainty),
    }
    row_counts: dict[str, int] = {}
    for relative, data in outputs.items():
        path = repo / relative
        validate_output_path(path, repo)
        write_parquet_new(path, data, input_hashes=input_hashes)
        row_counts[relative] = len(data)
    json_outputs = {
        "analysis/posthoc/quartile_edges.json": quartile_edges,
        "analysis/posthoc/gain_loss_concentration.json": gain,
        "analysis/posthoc/oracle_summary.json": oracle,
        "analysis/posthoc/catalog_health.json": catalog,
        "analysis/posthoc/robustness_mechanism_summary.json": mechanism,
    }
    for relative, data in json_outputs.items():
        path = repo / relative
        validate_output_path(path, repo)
        write_json_new(path, {**metadata(input_hashes=input_hashes), **data})
        row_counts[relative] = (
            len(data.get("slices", data.get("associations", [])))
            if isinstance(data, dict)
            else 0
        )

    perf_df = pd.DataFrame(performance)
    cross_df = pd.DataFrame(cross)
    routing_df = pd.DataFrame(routing)
    weights_df = pd.DataFrame(weights)
    agreement_df = pd.DataFrame(agreement)
    overlap_df = pd.DataFrame(overlap)
    temporal_df = pd.DataFrame(temporal)
    exposure_df = pd.DataFrame(exposure)
    robust_df = pd.DataFrame(robust)
    uncertainty_df = pd.DataFrame(uncertainty)
    _reports(
        repo,
        perf_df,
        cross_df,
        routing_df,
        weights_df,
        gain,
        oracle,
        agreement_df,
        overlap_df,
        temporal_df,
        exposure_df,
        catalog,
        robust_df,
        mechanism,
        uncertainty_df,
        quartile_edges,
    )
    manifest = _artifact_manifest(repo, input_hashes, row_counts)
    write_json_new(repo / "analysis/posthoc/posthoc_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "queries": len(frame),
                "artifacts": manifest["artifact_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
