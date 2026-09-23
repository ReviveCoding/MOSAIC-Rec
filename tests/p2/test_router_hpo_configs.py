from __future__ import annotations

from mosaic_rec.p2.router_hpo_configs import materialize_router_configs


def test_router_configs_use_selected_experts_and_alignment() -> None:
    hpo = {
        "protocol_id": "mosaic-rec-2026-09-23-p2",
        "selection_seed": 7,
        "epochs_per_trial": 10,
        "common": {"weight_decay": 0.01, "score_batch_size": 64},
        "families": {
            "R2": [{"trial_id": "R2_T1", "learning_rate": 0.1}],
            "R3": [{"trial_id": "R3_T1", "learning_rate": 0.2, "hidden": 16}],
            "R4": [
                {
                    "trial_id": "R4_T1",
                    "learning_rate": 0.3,
                    "hidden": 32,
                    "distillation_weight": 0.5,
                }
            ],
        },
    }
    ladder = {
        "protocol_id": hpo["protocol_id"],
        "common": {"batch_size": 8, "negatives_per_user": 2},
        "models": {
            "R2": {"gate": "monotonic", "hidden": 0, "distillation_weight": 0},
            "R3": {"gate": "context_mlp", "hidden": 8, "distillation_weight": 0},
            "R4": {"gate": "context_mlp", "hidden": 8, "distillation_weight": 1},
        },
    }
    selected = {
        "protocol_id": hpo["protocol_id"],
        "families": {
            "CSEQ_BERT4REC": {
                "selection": {
                    "status": "SELECTED_DEVELOPMENT",
                    "selected_trial_id": "S1",
                    "checkpoint": "/s.pt",
                }
            },
            "CCOLD_TWO_TOWER": {
                "selection": {
                    "status": "SELECTED_DEVELOPMENT",
                    "selected_trial_id": "C1",
                    "checkpoint": "/c.pt",
                }
            },
        },
    }
    alignment = {
        "protocol_id": hpo["protocol_id"],
        "selected_alignment_id": "A1",
        "selected_method": "temperature_scaling",
        "selected_temperatures": [0.5, 1.0],
    }
    configs = materialize_router_configs(
        hpo=hpo,
        ladder=ladder,
        pre_router_summary=selected,
        alignment_summary=alignment,
    )
    assert set(configs) == {"R2_T1", "R3_T1", "R4_T1"}
    assert configs["R3_T1"]["experts"]["sequence_checkpoint"] == "/s.pt"
    assert configs["R3_T1"]["common"]["alignment"] == "temperature_scaling"
    assert configs["R4_T1"]["models"]["R4"]["distillation_weight"] == 0.5
