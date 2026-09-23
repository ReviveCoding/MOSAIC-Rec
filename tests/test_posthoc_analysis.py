from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/posthoc_analyze_p1.py"
SPEC = importlib.util.spec_from_file_location("posthoc_analyze_p1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
posthoc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(posthoc)


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])
    first = posthoc.paired_bootstrap(values, seed=17, draws=200)
    second = posthoc.paired_bootstrap(values, seed=17, draws=200)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.all(first[0] <= values.mean(axis=0))
    assert np.all(first[1] >= values.mean(axis=0))


def test_cluster_bootstrap_is_deterministic() -> None:
    values = np.asarray([[0.0], [1.0], [2.0], [4.0]])
    clusters = np.asarray([1, 1, 2, 3])
    first = posthoc.paired_bootstrap(values, seed=23, draws=100, clusters=clusters)
    second = posthoc.paired_bootstrap(values, seed=23, draws=100, clusters=clusters)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_jensen_shannon_identity_and_symmetry() -> None:
    left = np.asarray([10, 20, 30])
    right = np.asarray([30, 20, 10])
    assert posthoc.jensen_shannon(left, left) == 0
    assert posthoc.jensen_shannon(left, right) == posthoc.jensen_shannon(right, left)


def test_read_role_requires_identical_keys(tmp_path: Path) -> None:
    base = pa.table(
        {
            "user_id": [1, 2],
            "item_id": [3, 4],
            "timestamp": [10, 20],
            "slice": ["controlled_cold", "warm_fit"],
            "rank_at_20": [1, None],
            "ndcg_at_20": [1.0, 0.0],
            "recall_at_20": [1.0, 0.0],
            "mrr_at_20": [1.0, 0.0],
        }
    )
    paths = {}
    for model in posthoc.MODELS:
        path = tmp_path / f"{model}.parquet"
        pq.write_table(base, path)
        paths[model] = path
    result = posthoc.read_role(paths)
    assert result["M1_hit_at_20"].tolist() == [1.0, 0.0]
    changed = base.set_column(0, "user_id", pa.array([1, 9]))
    pq.write_table(changed, paths["B4"])
    try:
        posthoc.read_role(paths)
    except ValueError as exc:
        assert "keys differ" in str(exc)
    else:
        raise AssertionError("Mismatched per-user keys were accepted")


def test_fixed_popularity_labels_preserve_zero() -> None:
    labels = posthoc.fixed_popularity_labels(
        np.asarray([0, 1, 5, 10]), np.asarray([2, 6]), "Q"
    )
    assert labels.tolist() == ["zero", "Q1", "Q2", "Q3"]


def test_operational_bins_are_upper_bound_inclusive() -> None:
    values = np.asarray([0, 1, 5, 6, 20, 21, 101])
    labels = posthoc.operational(
        values,
        [0, 5, 20, 50, 100],
        ["0", "1-5", "6-20", "21-50", "51-100", "101+"],
    )
    assert labels.tolist() == ["0", "1-5", "1-5", "6-20", "6-20", "21-50", "101+"]


def test_script_is_cpu_only_and_posthoc_guarded() -> None:
    source = SCRIPT.read_text()
    assert "import torch" not in source
    assert "score_locked" not in source
    assert '"POSTHOC_EXPLORATORY"' in source
    assert '"RETAIN_BASELINE"' in source


def test_manifest_verifier_checks_sources_and_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("immutable")
    output = tmp_path / "analysis"
    output.mkdir()
    table = output / "table.txt"
    table.write_text("derived")
    manifest = {
        "analysis_type": "POSTHOC_EXPLORATORY",
        "confirmatory": False,
        "may_change_p1_decision": False,
        "p1_decision": "RETAIN_BASELINE",
        "sources": {"source": {"path": str(source), "sha256": posthoc.sha256(source)}},
        "outputs": {"table.txt": posthoc.sha256(table)},
        "population": {"locked_users": 1},
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest))
    assert posthoc.verify_manifest(output)["status"] == "PASS"
    source.write_text("changed")
    try:
        posthoc.verify_manifest(output)
    except RuntimeError as exc:
        assert "source" in str(exc)
    else:
        raise AssertionError("Changed immutable source was accepted")
