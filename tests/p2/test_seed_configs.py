from __future__ import annotations

import json
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file
from mosaic_rec.p2.seed_configs import materialize_seed_configs


def test_seed_configs_pair_router_with_same_seed_experts(tmp_path: Path) -> None:
    protocol = "mosaic-rec-2026-09-23-p2"
    families = (
        "CSEQ_BERT4REC",
        "CCOLD_TWO_TOWER",
        "LEGACY_M1",
        "BPR_MF",
        "LIGHTGCN",
        "SASREC",
        "VBPR",
        "R2",
        "R3",
        "R4",
    )
    trials = []
    selections = {}
    for family in families:
        trial_id = f"{family}_T1"
        path = tmp_path / f"{trial_id}.json"
        if family == "CSEQ_BERT4REC":
            config = {"sequence": {"model_id": trial_id}}
        elif family == "CCOLD_TWO_TOWER":
            config = {"content": {"model_id": trial_id}}
        elif family == "LEGACY_M1":
            config = {"model_id": trial_id}
        else:
            config = {"models": {family: {"model_id": trial_id}}}
        config.update({"protocol_id": protocol, "family": family})
        path.write_text(json.dumps(config))
        trials.append(
            {
                "trial_id": trial_id,
                "family": family,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
        selections[family] = {
            "selection": {
                "status": "SELECTED_DEVELOPMENT",
                "selected_trial_id": trial_id,
            }
        }
    configs = materialize_seed_configs(
        repo=tmp_path,
        scratch=Path("/scratch"),
        hpo={"protocol_id": protocol, "independent_seeds": [11, 12]},
        pre_router_manifest={
            "protocol_id": protocol,
            "trials": [row for row in trials if not row["family"].startswith("R")],
        },
        pre_router_summary={
            "protocol_id": protocol,
            "families": {
                key: selections[key] for key in selections if not key.startswith("R")
            },
        },
        router_manifest={
            "protocol_id": protocol,
            "trials": [row for row in trials if row["family"].startswith("R")],
        },
        router_summary={
            "protocol_id": protocol,
            "families": {
                key: selections[key] for key in selections if key.startswith("R")
            },
        },
    )
    assert len(configs) == 20
    router = configs["E01-R4-SEED11"]
    assert router["experts"]["sequence_checkpoint"].endswith(
        "e01-cseq_bert4rec-seed11.pt"
    )
    assert router["experts"]["content_checkpoint"].endswith(
        "e01-ccold_two_tower-seed11.pt"
    )
    assert configs["E01-CSEQ_BERT4REC-SEED12"]["sequence"]["model_id"].endswith(
        "SEED12"
    )
