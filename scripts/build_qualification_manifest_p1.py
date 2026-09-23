"""Freeze E02-E04 qualification inputs and selection rules before access."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / "evidence/e01_seed_qualification_p1.json"
RESCORE = ROOT / "evidence/e01_sequence_rescore_p1.json"
SPLIT = ROOT / "evidence/microlens_split_p1.json"
OUTPUT = ROOT / "configs/qualification_selection_p1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Qualification manifest exists; preserve the frozen file")
    e01 = json.loads(E01.read_text())
    rescore = json.loads(RESCORE.read_text())
    split = json.loads(SPLIT.read_text())
    if e01["status"] != "E01_SEED_QUALIFICATION_COMPLETE_UNQUALIFIED":
        raise ValueError("E01 is not terminal")
    if rescore["status"] != "SEQUENCE_COMMON_SLICE_RESCORE_PASS":
        raise ValueError("Sequence replay gate is not satisfied")
    if len(e01["seeds"]) != 3:
        raise ValueError("Qualification requires exactly three frozen seed checkpoints")

    architecture = {
        "B7_BERT4REC_T3": ("bert4rec", None),
        "B4_TWO_TOWER_T3": ("two_tower_content", None),
        "M1_T1": ("mosaic", 1),
        "M1_T2": ("mosaic", 1),
        "M4_T1": ("mosaic", 4),
        "M4_T3": ("mosaic", 4),
        "M6_T3": ("mosaic", 6),
    }
    candidates = []
    for config_id, summary in e01["candidates"].items():
        checkpoints = []
        for row in summary["seed_receipts"]:
            receipt = ROOT / "evidence" / f"{row['receipt_id']}.json"
            source = json.loads(receipt.read_text())
            checkpoint = Path(source["checkpoint"])
            if digest(checkpoint) != row["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {config_id}")
            checkpoints.append(
                {
                    "seed": row["seed"],
                    "receipt": str(receipt.relative_to(ROOT)).replace("\\", "/"),
                    "receipt_sha256": digest(receipt),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )
        model_architecture, variant = architecture[config_id]
        candidates.append(
            {
                "config_id": config_id,
                "architecture": model_architecture,
                "variant": variant,
                "learning_rate": summary["learning_rate"],
                "epochs": summary["epochs"],
                "checkpoints": checkpoints,
            }
        )

    manifest = {
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "manifest_id": "qualification-selection-p1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "state_transition": {
            "from": "DEVELOPMENT",
            "to": "QUALIFICATION",
            "authorized_roles": ["QUALIFICATION"],
            "prohibited_roles": ["LOCKED_FINAL"],
        },
        "preconditions": {
            "e01_summary": str(E01.relative_to(ROOT)).replace("\\", "/"),
            "e01_summary_sha256": digest(E01),
            "sequence_rescore_summary": str(RESCORE.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "sequence_rescore_summary_sha256": digest(RESCORE),
            "canonical_receipts_before_qualification": 141,
            "locked_execution_receipts": 0,
        },
        "data": {
            "dataset": "microlens-100k",
            "fit_visible": split["outputs"]["FIT_VISIBLE"],
            "qualification": split["outputs"]["QUALIFICATION"],
            "controlled_cold_ids": {
                "path": "/home/bjw-0/mosaic-rec-work/data/microlens-100k/split_p1/controlled_cold_item_ids.txt",
                "sha256": split["cold_cohort_sha256"],
                "count": split["cold_cohort_count"],
            },
            "locked_final_sha256_recorded_but_access_prohibited": split["outputs"][
                "LOCKED_FINAL"
            ]["sha256"],
        },
        "evaluation": {
            "ranking": "exact_full_catalog_cuda",
            "ensemble": "arithmetic_mean_of_three_seed_checkpoint_scores_before_ranking",
            "query": "first_QUALIFICATION_event_per_user_with_FIT_VISIBLE_history",
            "history": "FIT_VISIBLE_only",
            "seen_exclusion": "all_user_FIT_VISIBLE_items",
            "candidate_availability": "source_first_observed_timestamp_lte_query_timestamp",
            "tie_break": "ascending_item_id_via_negative_1e-12_item_index_offset",
            "primary_metric": "ndcg_at_20",
            "secondary_metrics": ["recall_at_20", "mrr_at_20"],
            "slices": ["all", "controlled_cold", "warm_fit"],
            "per_user_rows_required": True,
            "cuda_required": True,
        },
        "candidates": candidates,
        "selection": {
            "C_seq": {
                "eligible": ["B7_BERT4REC_T3"],
                "rule": "freeze after valid qualification ensemble scoring; ID-only model is ineligible for C_cold",
            },
            "C_cold": {
                "eligible": ["B4_TWO_TOWER_T3"],
                "rule": "freeze after valid qualification ensemble scoring; content remains visible and controlled-cold collaborative support remains absent",
            },
            "MOSAIC_finalist": {
                "eligible": ["M1_T1", "M1_T2", "M4_T1", "M4_T3", "M6_T3"],
                "rule": "minimize the worse of controlled_cold and warm_fit ordinal ranks by qualification NDCG@20; tie-break by higher all NDCG@20, then lexicographic config_id",
                "rank_direction": "higher_is_better",
                "no_seed_selection": True,
            },
        },
        "e04_predeclared_rules": {
            "warm_noninferiority_margin": "0.10 * C_seq qualification warm_fit NDCG@20",
            "margin_scale": "absolute_ndcg_at_20",
            "primary_bootstrap": "paired_user_cluster",
            "cold_sensitivity_bootstrap": "item_cluster",
            "bootstrap_draws": 10000,
            "secondary_family_control": "Holm",
            "support_bins": {
                "cold": 0,
                "emerging": [1, 5],
                "sparse": [6, 20],
                "warm_min": 21,
            },
            "robustness_family": [
                "image_missing",
                "text_missing",
                "metadata_missing",
                "image_embedding_corruption",
                "mismatched_image",
                "text_truncation",
                "collaborative_support_reduction",
                "temporal_shift",
                "long_tail",
            ],
            "confirmatory_ablation_family": [
                "no_sequence",
                "no_collaborative",
                "no_text",
                "no_image",
                "no_metadata",
                "no_adaptive_gate",
                "no_behavioral_dropout",
                "no_modality_dropout",
                "no_consistency_signal",
                "unconstrained_vs_monotonic_gate",
                "no_auxiliary_alignment",
                "retrieval_only_vs_reranking",
            ],
        },
        "guards": [
            "No hyperparameter, candidate, seed, metric, slice, or selection-rule changes after QUALIFICATION access under p1.",
            "No LOCKED_FINAL file read before E05 is complete and the FROZEN guard passes.",
            "A failed qualification candidate remains evidence and receives no replacement seed.",
        ],
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
