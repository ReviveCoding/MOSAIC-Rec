"""Materialize independent-seed configs from frozen p2 HPO selections."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import canonical_json_sha256, sha256_file, write_json_new
from mosaic_rec.p2.hpo_summary import PRE_ROUTER_FAMILIES
from mosaic_rec.p2.router_hpo_configs import ROUTER_FAMILIES


def _set_model_identity(config: dict[str, Any], family: str, identity: str) -> None:
    if family == "CSEQ_BERT4REC":
        config["sequence"]["model_id"] = identity
    elif family == "CCOLD_TWO_TOWER":
        config["content"]["model_id"] = identity
    elif family == "LEGACY_M1":
        config["model_id"] = identity
    elif family in {"BPR_MF", "LIGHTGCN", "SASREC", "VBPR"}:
        config["models"][family]["model_id"] = identity
        config["models"][family]["stage_id"] = "P2-E01"
    elif family in ROUTER_FAMILIES:
        config["models"][family]["model_id"] = identity
    else:
        raise ValueError(f"Unknown p2 family: {family}")


def materialize_seed_configs(
    *,
    repo: Path,
    scratch: Path,
    hpo: dict[str, Any],
    pre_router_manifest: dict[str, Any],
    pre_router_summary: dict[str, Any],
    router_manifest: dict[str, Any],
    router_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    sources = (
        hpo,
        pre_router_manifest,
        pre_router_summary,
        router_manifest,
        router_summary,
    )
    if any(source.get("protocol_id") != PROTOCOL_ID for source in sources):
        raise RuntimeError("Independent-seed protocol mismatch")
    identities = {
        trial["trial_id"]: trial
        for trial in pre_router_manifest["trials"] + router_manifest["trials"]
    }
    selections = {
        family: pre_router_summary["families"][family]["selection"]
        for family in PRE_ROUTER_FAMILIES
    }
    selections.update(
        {
            family: router_summary["families"][family]["selection"]
            for family in ROUTER_FAMILIES
        }
    )
    output: dict[str, dict[str, Any]] = {}
    for seed in hpo["independent_seeds"]:
        seed = int(seed)
        expert_paths = {
            "sequence_checkpoint": str(
                scratch
                / f"seed_qualification/checkpoints/e01-cseq_bert4rec-seed{seed}.pt"
            ),
            "content_checkpoint": str(
                scratch
                / f"seed_qualification/checkpoints/e01-ccold_two_tower-seed{seed}.pt"
            ),
        }
        for family, selection in selections.items():
            if selection["status"] != "SELECTED_DEVELOPMENT":
                raise RuntimeError(f"Family is not seed-qualifiable: {family}")
            source_id = selection["selected_trial_id"]
            identity = identities[source_id]
            source_path = Path(identity["path"])
            if not source_path.is_absolute():
                source_path = repo / source_path
            if sha256_file(source_path) != identity["sha256"]:
                raise RuntimeError(f"Selected config hash mismatch: {source_id}")
            config = copy.deepcopy(json.loads(source_path.read_text()))
            run_id = f"E01-{family}-SEED{seed}"
            config.update(
                {
                    "stage_id": "P2-E01",
                    "trial_id": run_id,
                    "family": family,
                    "seed": seed,
                    "selected_hpo_trial_id": source_id,
                    "selected_hpo_config_sha256": identity["sha256"],
                    "evidence_class": "p2_measured_development_independent_seed",
                    "protected_roles_opened": [],
                }
            )
            _set_model_identity(config, family, run_id)
            if family in ROUTER_FAMILIES:
                config["experts"] = expert_paths
            output[run_id] = config
    return output


def write_seed_configs(
    *,
    output_dir: Path,
    configs: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id, config in sorted(configs.items()):
        path = output_dir / f"{run_id.lower()}.json"
        write_json_new(path, config)
        rows.append(
            {
                "run_id": run_id,
                "family": config["family"],
                "seed": config["seed"],
                "path": str(path),
                "sha256": sha256_file(path),
                "canonical_config_sha256": canonical_json_sha256(config),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E01-CONFIG-MATERIALIZATION",
        "status": "PASS",
        "source_hashes": source_hashes,
        "run_count": len(rows),
        "runs": rows,
        "protected_roles_opened": [],
    }
