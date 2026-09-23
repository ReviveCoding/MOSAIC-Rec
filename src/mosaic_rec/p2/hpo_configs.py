"""Mechanical materialization of the frozen p2 HPO search space."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import canonical_json_sha256, sha256_file, write_json_new

EXPERT_FAMILIES = {"CSEQ_BERT4REC": "sequence", "CCOLD_TWO_TOWER": "content"}
ADDITIONAL_FAMILIES = {"BPR_MF", "LIGHTGCN", "SASREC", "VBPR"}


def _trial_common(hpo: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    common = copy.deepcopy(hpo["common"])
    common.update(
        {
            "epochs": int(hpo["epochs_per_trial"]),
            "learning_rate": float(trial["learning_rate"]),
        }
    )
    return common


def materialize_open_hpo_configs(
    *,
    hpo: dict[str, Any],
    development: dict[str, Any],
    additional: dict[str, Any],
    legacy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if hpo["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("HPO protocol mismatch")
    output: dict[str, dict[str, Any]] = {}
    for family, section_name in EXPERT_FAMILIES.items():
        for trial in hpo["families"][family]:
            section = copy.deepcopy(development[section_name])
            section.update(_trial_common(hpo, trial))
            section["model_id"] = trial["trial_id"]
            output[trial["trial_id"]] = {
                "protocol_id": PROTOCOL_ID,
                "stage_id": "P2-E00",
                "trial_id": trial["trial_id"],
                "family": family,
                "seed": int(hpo["selection_seed"]),
                section_name: section,
                "scoring": copy.deepcopy(development["scoring"]),
                "evidence_class": "p2_measured_development_hpo",
                "protected_roles_opened": [],
            }
    for family in sorted(ADDITIONAL_FAMILIES):
        for trial in hpo["families"][family]:
            model = copy.deepcopy(additional["models"][family])
            model["model_id"] = trial["trial_id"]
            model["stage_id"] = "P2-E00"
            output[trial["trial_id"]] = {
                "protocol_id": PROTOCOL_ID,
                "stage": "P2-E00",
                "trial_id": trial["trial_id"],
                "family": family,
                "seed": int(hpo["selection_seed"]),
                "common": {
                    **copy.deepcopy(additional["common"]),
                    **_trial_common(hpo, trial),
                },
                "models": {family: model},
                "evidence_class": "p2_measured_development_hpo",
                "protected_roles_opened": [],
            }
    for trial in hpo["families"]["LEGACY_M1"]:
        config = copy.deepcopy(legacy)
        config.update(_trial_common(hpo, trial))
        config.update(
            {
                "stage_id": "P2-E00",
                "model_id": trial["trial_id"],
                "trial_id": trial["trial_id"],
                "family": "LEGACY_M1",
                "seed": int(hpo["selection_seed"]),
                "evidence_class": "p2_measured_development_hpo",
            }
        )
        output[trial["trial_id"]] = config
    return output


def write_materialized_configs(
    *, output_dir: Path, configs: dict[str, dict[str, Any]], hpo_path: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial_id, config in sorted(configs.items()):
        path = output_dir / f"{trial_id.lower()}.json"
        write_json_new(path, config)
        rows.append(
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
        "stage_id": "P2-E00-CONFIG-MATERIALIZATION",
        "status": "PASS",
        "hpo_manifest_sha256": sha256_file(hpo_path),
        "trial_count": len(rows),
        "trials": rows,
        "protected_roles_opened": [],
    }
