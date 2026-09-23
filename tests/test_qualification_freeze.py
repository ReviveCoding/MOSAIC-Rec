from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "evidence" / name).read_text())


def test_qualification_freeze_is_internally_consistent() -> None:
    e02 = load("e02_comparator_freeze_p1.json")
    e03 = load("e03_mosaic_freeze_p1.json")
    e04 = load("e04_scientific_freeze_p1.json")
    e05 = load("e05_freeze_manifest_p1.json")
    assert e02["C_seq"]["config_id"] == "B7_BERT4REC_T3"
    assert e02["C_cold"]["config_id"] == "B4_TWO_TOWER_T3"
    assert e03["finalist"] == "M1_T2"
    assert e04["warm_noninferiority_margin"] == (
        0.10 * e02["C_seq"]["qualification_slices"]["warm_fit"]["ndcg_at_20"]
    )
    assert e05["C_seq"] == e02["C_seq"]["config_id"]
    assert e05["C_cold"] == e02["C_cold"]["config_id"]
    assert e05["MOSAIC_finalist"] == e03["finalist"]
    for stage, filename in (
        ("E02", "e02_comparator_freeze_p1.json"),
        ("E03", "e03_mosaic_freeze_p1.json"),
        ("E04", "e04_scientific_freeze_p1.json"),
    ):
        value = hashlib.sha256((ROOT / "evidence" / filename).read_bytes()).hexdigest()
        assert e05["source_freeze_receipts"][stage]["sha256"] == value
