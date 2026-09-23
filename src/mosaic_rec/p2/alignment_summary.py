"""Select the p2 score alignment using the frozen VALIDATION criterion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new

METHOD_ORDER = {
    "temperature_scaling": 0,
    "per_query_standardized_logits": 1,
    "rank_percentile_normalization": 2,
}


def alignment_score(
    *, cold: float, warm: float, cold_comparator: float, warm_comparator: float
) -> dict[str, float]:
    margin = 0.10 * warm_comparator
    if margin <= 0:
        raise RuntimeError("C_seq warm VALIDATION NDCG@20 must be positive")
    cold_unit = (
        cold / cold_comparator if cold_comparator > 0 else cold - cold_comparator
    )
    warm_unit = 1.0 + (warm - warm_comparator) / margin
    return {
        "cold_unit": cold_unit,
        "warm_unit": warm_unit,
        "selection_score": 0.5 * (cold_unit + warm_unit),
        "warm_margin": margin,
    }


def summarize_alignment(
    *, repo: Path, manifest_path: Path, implementation_path: Path, output: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    implementation = json.loads(implementation_path.read_text())
    if (
        manifest["protocol_id"] != PROTOCOL_ID
        or implementation["protocol_id"] != PROTOCOL_ID
    ):
        raise RuntimeError("Alignment protocol mismatch")
    cseq = json.loads(
        (repo / "evidence/p2/p2-cseq-bert4rec-pilot-attempt2.json").read_text()
    )
    ccold = json.loads(
        (repo / "evidence/p2/p2-ccold-two-tower-pilot-attempt2.json").read_text()
    )
    warm_comparator = float(cseq["validation"]["slices"]["warm"]["ndcg_at_20"])
    cold_comparator = float(
        ccold["validation"]["slices"]["controlled_zero"]["ndcg_at_20"]
    )
    trials = []
    for identity in manifest["trials"]:
        alignment_id = identity["alignment_id"]
        config_path = repo / identity["path"]
        if sha256_file(config_path) != identity["sha256"]:
            raise RuntimeError(f"Alignment config hash mismatch: {alignment_id}")
        config = json.loads(config_path.read_text())
        receipt_path = repo / f"evidence/p2/p2-b03-{alignment_id.lower()}.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"Missing alignment receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "PASS":
            raise RuntimeError(f"Alignment trial did not pass: {receipt_path}")
        scored = Path(receipt["validation"]["output"])
        if sha256_file(scored) != receipt["validation"]["output_sha256"]:
            raise RuntimeError(f"Alignment output hash mismatch: {receipt_path}")
        slices = receipt["validation"]["slices"]
        score = alignment_score(
            cold=float(slices["controlled_zero"]["ndcg_at_20"] or 0.0),
            warm=float(slices["warm"]["ndcg_at_20"] or 0.0),
            cold_comparator=cold_comparator,
            warm_comparator=warm_comparator,
        )
        trials.append(
            {
                "alignment_id": alignment_id,
                "method": config["router"]["alignment"],
                "temperatures": config["router"].get("temperatures"),
                "validation": slices,
                **score,
            }
        )
    best_score = max(trial["selection_score"] for trial in trials)
    tied = [trial for trial in trials if trial["selection_score"] == best_score]
    selected = min(
        tied,
        key=lambda trial: (METHOD_ORDER[trial["method"]], trial["alignment_id"]),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-B03-SCORE-ALIGNMENT",
        "status": "PASS_DEVELOPMENT",
        "evidence_class": "p2_measured_development_score_alignment_summary",
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": sha256_file(implementation_path),
        "cseq_receipt_sha256": sha256_file(
            repo / "evidence/p2/p2-cseq-bert4rec-pilot-attempt2.json"
        ),
        "ccold_receipt_sha256": sha256_file(
            repo / "evidence/p2/p2-ccold-two-tower-pilot-attempt2.json"
        ),
        "trials": trials,
        "selected_alignment_id": selected["alignment_id"],
        "selected_method": selected["method"],
        "selected_temperatures": selected["temperatures"],
        "protected_roles_opened": [],
    }
    write_json_new(output, result)
    return result
