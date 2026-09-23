"""Materialize router HPO trials after open-data experts and alignment are selected."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import canonical_json_sha256, sha256_file, write_json_new

ROUTER_FAMILIES = ("R2", "R3", "R4")


def materialize_router_configs(
    *,
    hpo: dict[str, Any],
    ladder: dict[str, Any],
    pre_router_summary: dict[str, Any],
    alignment_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    sources = (hpo, ladder, pre_router_summary, alignment_summary)
    if any(source.get("protocol_id") != PROTOCOL_ID for source in sources):
        raise RuntimeError("Router HPO protocol mismatch")
    sequence = pre_router_summary["families"]["CSEQ_BERT4REC"]["selection"]
    content = pre_router_summary["families"]["CCOLD_TWO_TOWER"]["selection"]
    if (
        sequence["status"] != "SELECTED_DEVELOPMENT"
        or content["status"] != "SELECTED_DEVELOPMENT"
    ):
        raise RuntimeError("Router HPO requires selected open-data experts")
    method = alignment_summary["selected_method"]
    temperatures = alignment_summary.get("selected_temperatures") or [1.0, 1.0]
    output: dict[str, dict[str, Any]] = {}
    for family in ROUTER_FAMILIES:
        for trial in hpo["families"][family]:
            common = copy.deepcopy(ladder["common"])
            common.update(
                {
                    "epochs": int(hpo["epochs_per_trial"]),
                    "learning_rate": float(trial["learning_rate"]),
                    "weight_decay": float(hpo["common"]["weight_decay"]),
                    "score_batch_size": int(hpo["common"]["score_batch_size"]),
                    "alignment": method,
                    "temperatures": temperatures,
                }
            )
            model = copy.deepcopy(ladder["models"][family])
            model["model_id"] = trial["trial_id"]
            if "hidden" in trial:
                model["hidden"] = int(trial["hidden"])
            if "distillation_weight" in trial:
                model["distillation_weight"] = float(trial["distillation_weight"])
            output[trial["trial_id"]] = {
                "protocol_id": PROTOCOL_ID,
                "stage_id": "P2-E00",
                "trial_id": trial["trial_id"],
                "family": family,
                "seed": int(hpo["selection_seed"]),
                "common": common,
                "models": {family: model},
                "experts": {
                    "sequence_checkpoint": sequence["checkpoint"],
                    "content_checkpoint": content["checkpoint"],
                },
                "selected_expert_trials": {
                    "sequence": sequence["selected_trial_id"],
                    "content": content["selected_trial_id"],
                },
                "selected_alignment_id": alignment_summary["selected_alignment_id"],
                "evidence_class": "p2_measured_development_hpo",
                "protected_roles_opened": [],
            }
    return output


def write_router_configs(
    *,
    output_dir: Path,
    configs: dict[str, dict[str, Any]],
    hpo_path: Path,
    pre_router_summary_path: Path,
    alignment_summary_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for trial_id, config in sorted(configs.items()):
        path = output_dir / f"{trial_id.lower()}.json"
        write_json_new(path, config)
        trials.append(
            {
                "trial_id": trial_id,
                "family": config["family"],
                "path": str(path),
                "sha256": sha256_file(path),
                "canonical_config_sha256": canonical_json_sha256(config),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E00-ROUTER-CONFIG-MATERIALIZATION",
        "status": "PASS",
        "hpo_sha256": sha256_file(hpo_path),
        "pre_router_summary_sha256": sha256_file(pre_router_summary_path),
        "alignment_summary_sha256": sha256_file(alignment_summary_path),
        "trial_count": len(trials),
        "trials": trials,
        "protected_roles_opened": [],
    }
