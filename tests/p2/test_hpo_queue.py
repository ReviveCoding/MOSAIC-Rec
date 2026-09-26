from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module(repo: Path):  # type: ignore[no-untyped-def]
    path = repo / "scripts/p2/run_hpo_queue.py"
    spec = importlib.util.spec_from_file_location("run_hpo_queue", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_validation_is_hash_strict(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    module = _module(repo)
    paths = module.trial_paths(tmp_path, tmp_path / "scratch", "CSEQ_T1")
    paths["checkpoint"].parent.mkdir(parents=True)
    paths["output"].parent.mkdir(parents=True)
    paths["receipt"].parent.mkdir(parents=True)
    paths["checkpoint"].write_bytes(b"checkpoint")
    paths["output"].write_bytes(b"output")
    receipt = {
        "status": "PASS",
        "config_sha256": "config-hash",
        "checkpoint_sha256": module.sha256_file(paths["checkpoint"]),
        "validation": {"output_sha256": module.sha256_file(paths["output"])},
    }
    paths["receipt"].write_text(json.dumps(receipt))
    assert module.validate_terminal(paths, "config-hash") == receipt
    paths["output"].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="output hash mismatch"):
        module.validate_terminal(paths, "config-hash")


def test_manifest_config_hash_uses_materialized_file_hash(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    module = _module(repo)
    assert module.manifest_config_hash({"sha256": "file-hash"}) == "file-hash"
    with pytest.raises(RuntimeError, match="missing sha256"):
        module.manifest_config_hash({"canonical_config_sha256": "canonical-hash"})


def test_family_runner_mapping(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    module = _module(repo)
    common = {
        "trial_id": "X",
        "repo": repo,
        "scratch": tmp_path,
        "lock": tmp_path / "l",
    }
    expert = module.command_for(family="CSEQ_BERT4REC", **common)
    legacy = module.command_for(family="LEGACY_M1", **common)
    baseline = module.command_for(family="VBPR", **common)
    assert expert[1].endswith("run_expert_hpo_trial.py")
    assert legacy[1].endswith("run_legacy_hpo_trial.py")
    assert baseline[1].endswith("run_baseline_hpo_trial.py")


def test_audited_retry_uses_distinct_log_only(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    module = _module(repo)
    paths = module.trial_paths(repo, tmp_path, "LIGHTGCN_T3")
    assert paths["log"].name == "lightgcn_t3_attempt2.log"
    assert paths["checkpoint"].name == "lightgcn_t3.pt"
    assert paths["output"].name == "lightgcn_t3_validation.parquet"
    assert paths["receipt"].name == "p2-e00-lightgcn_t3.json"
