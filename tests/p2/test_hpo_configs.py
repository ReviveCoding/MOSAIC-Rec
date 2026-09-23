from __future__ import annotations

import json
from pathlib import Path

from mosaic_rec.p2.hpo_configs import (
    materialize_open_hpo_configs,
    validate_trial_identity,
)

ROOT = Path(__file__).parents[2]


def test_frozen_open_hpo_trials_materialize_without_scientific_changes() -> None:
    configs = materialize_open_hpo_configs(
        hpo=json.loads((ROOT / "configs/p2/hpo.json").read_text()),
        development=json.loads(
            (ROOT / "configs/p2/development_pilots_attempt2.json").read_text()
        ),
        additional=json.loads(
            (ROOT / "configs/p2/additional_baseline_pilots.json").read_text()
        ),
        legacy=json.loads((ROOT / "configs/p2/legacy_m1_pilot.json").read_text()),
    )
    assert len(configs) == 21
    assert len({config["trial_id"] for config in configs.values()}) == 21
    assert {config["seed"] for config in configs.values()} == {20260930}
    assert all(
        config.get("protected_roles_opened") == [] for config in configs.values()
    )
    assert all(
        config.get("common", config).get("epochs") == 10
        or next(config[key] for key in ("sequence", "content") if key in config)[
            "epochs"
        ]
        == 10
        for config in configs.values()
    )


def test_committed_trial_manifest_validates_config_and_hpo_hashes() -> None:
    manifest = json.loads((ROOT / "configs/p2/hpo_trials_manifest.json").read_text())
    row = validate_trial_identity(
        manifest=manifest,
        trial_id="CSEQ_T1",
        config_path=ROOT / "configs/p2/hpo_trials/cseq_t1.json",
        hpo_path=ROOT / "configs/p2/hpo.json",
    )
    assert row["family"] == "CSEQ_BERT4REC"
