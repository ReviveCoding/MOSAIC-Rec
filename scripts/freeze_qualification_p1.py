"""Apply frozen E02-E04 rules and build the complete E05 manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qualification_selection_p1.json"
OUTPUTS = {
    "E02": ROOT / "evidence/e02_comparator_freeze_p1.json",
    "E03": ROOT / "evidence/e03_mosaic_freeze_p1.json",
    "E04": ROOT / "evidence/e04_scientific_freeze_p1.json",
    "E05": ROOT / "evidence/e05_freeze_manifest_p1.json",
}
RECEIPTS = {
    "B7_BERT4REC_T3": ROOT / "evidence/e02_qualification_p1_b7_bert4rec_t3.json",
    "B4_TWO_TOWER_T3": ROOT / "evidence/e02_qualification_p1_b4_two_tower_t3.json",
    "M1_T1": ROOT / "evidence/e03_qualification_p1_m1_t1.json",
    "M1_T2": ROOT / "evidence/e03_qualification_p1_m1_t2.json",
    "M4_T1": ROOT / "evidence/e03_qualification_p1_m4_t1.json",
    "M4_T3": ROOT / "evidence/e03_qualification_p1_m4_t3.json",
    "M6_T3": ROOT / "evidence/e03_qualification_p1_m6_t3.json",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def load_receipts(manifest_hash: str) -> dict[str, dict]:
    results = {}
    for config_id, path in RECEIPTS.items():
        receipt = json.loads(path.read_text())
        if (
            receipt["status"] != "QUALIFICATION_SCORE_PASS"
            or receipt["state"] != "QUALIFICATION"
            or receipt["role"] != "QUALIFICATION"
            or receipt["model"] != config_id
            or receipt["manifest_sha256"] != manifest_hash
            or not receipt["checkpoint_unchanged"]
        ):
            raise ValueError(f"Invalid qualification receipt: {config_id}")
        per_user = Path(receipt["per_user_path"])
        if digest(per_user) != receipt["per_user_sha256"]:
            raise ValueError(f"Per-user hash mismatch: {config_id}")
        results[config_id] = receipt
    return results


def ndcg(receipt: dict, slice_name: str) -> float:
    return float(receipt["validation_slices"][slice_name]["ndcg_at_20"])


def ordinal_ranks(values: dict[str, float]) -> dict[str, int]:
    unique = sorted(set(values.values()), reverse=True)
    return {name: unique.index(value) + 1 for name, value in values.items()}


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    if any(path.exists() for path in OUTPUTS.values()):
        raise FileExistsError("Freeze output exists; preserve the frozen evidence")
    status = json.loads((ROOT / "evidence/stage_status.json").read_text())
    if status["scientific_state"] != "QUALIFICATION":
        raise RuntimeError("Protocol is not in QUALIFICATION")
    manifest = json.loads(CONFIG.read_text())
    manifest_hash = digest(CONFIG)
    results = load_receipts(manifest_hash)
    now = datetime.now(timezone.utc).isoformat()
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    c_seq = manifest["selection"]["C_seq"]["eligible"]
    c_cold = manifest["selection"]["C_cold"]["eligible"]
    if c_seq != ["B7_BERT4REC_T3"] or c_cold != ["B4_TWO_TOWER_T3"]:
        raise ValueError("Comparator eligibility changed")
    e02 = {
        "stage_id": "E02",
        "protocol_id": manifest["protocol_id"],
        "state": "QUALIFICATION",
        "status": "COMPARATORS_FROZEN",
        "ended_utc": now,
        "git_commit": git_commit,
        "manifest_sha256": manifest_hash,
        "C_seq": {
            "config_id": c_seq[0],
            "qualification_receipt": str(RECEIPTS[c_seq[0]].relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "qualification_receipt_sha256": digest(RECEIPTS[c_seq[0]]),
            "qualification_slices": results[c_seq[0]]["validation_slices"],
        },
        "C_cold": {
            "config_id": c_cold[0],
            "qualification_receipt": str(RECEIPTS[c_cold[0]].relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "qualification_receipt_sha256": digest(RECEIPTS[c_cold[0]]),
            "qualification_slices": results[c_cold[0]]["validation_slices"],
        },
        "warnings": [
            "B7 is ID-only and is not the cold comparator.",
            "Comparator identities cannot change after this freeze under p1.",
        ],
        "evidence_class": "derived_qualification_freeze",
    }
    write(OUTPUTS["E02"], e02)

    eligible = manifest["selection"]["MOSAIC_finalist"]["eligible"]
    cold_values = {name: ndcg(results[name], "controlled_cold") for name in eligible}
    warm_values = {name: ndcg(results[name], "warm_fit") for name in eligible}
    overall_values = {name: ndcg(results[name], "all") for name in eligible}
    cold_ranks = ordinal_ranks(cold_values)
    warm_ranks = ordinal_ranks(warm_values)
    ordered = sorted(
        eligible,
        key=lambda name: (
            max(cold_ranks[name], warm_ranks[name]),
            -overall_values[name],
            name,
        ),
    )
    finalist = ordered[0]
    rank_table = [
        {
            "config_id": name,
            "controlled_cold_ndcg_at_20": cold_values[name],
            "controlled_cold_rank": cold_ranks[name],
            "warm_fit_ndcg_at_20": warm_values[name],
            "warm_fit_rank": warm_ranks[name],
            "worst_rank": max(cold_ranks[name], warm_ranks[name]),
            "all_ndcg_at_20": overall_values[name],
        }
        for name in eligible
    ]
    e03 = {
        "stage_id": "E03",
        "protocol_id": manifest["protocol_id"],
        "state": "QUALIFICATION",
        "status": "MOSAIC_FINALIST_FROZEN",
        "ended_utc": now,
        "git_commit": git_commit,
        "manifest_sha256": manifest_hash,
        "selection_rule": manifest["selection"]["MOSAIC_finalist"]["rule"],
        "rank_table": rank_table,
        "finalist": finalist,
        "finalist_qualification_receipt": str(
            RECEIPTS[finalist].relative_to(ROOT)
        ).replace("\\", "/"),
        "finalist_qualification_receipt_sha256": digest(RECEIPTS[finalist]),
        "warnings": [
            "M4_T3 cold strength and warm degradation are preserved in the rank table.",
            "Finalist identity cannot change after this freeze under p1.",
        ],
        "evidence_class": "derived_qualification_freeze",
    }
    write(OUTPUTS["E03"], e03)

    c_seq_warm = ndcg(results[c_seq[0]], "warm_fit")
    margin = 0.10 * c_seq_warm
    e04 = {
        "stage_id": "E04",
        "protocol_id": manifest["protocol_id"],
        "state": "QUALIFICATION",
        "status": "MARGIN_STATISTICS_ROBUSTNESS_ABLATION_FROZEN",
        "ended_utc": now,
        "git_commit": git_commit,
        "manifest_sha256": manifest_hash,
        "warm_noninferiority_margin": margin,
        "margin_derivation": {
            "formula": manifest["e04_predeclared_rules"]["warm_noninferiority_margin"],
            "C_seq_warm_fit_ndcg_at_20": c_seq_warm,
        },
        "statistics": {
            key: manifest["e04_predeclared_rules"][key]
            for key in (
                "primary_bootstrap",
                "cold_sensitivity_bootstrap",
                "bootstrap_draws",
                "secondary_family_control",
                "support_bins",
            )
        },
        "robustness_family": manifest["e04_predeclared_rules"]["robustness_family"],
        "confirmatory_ablation_family": manifest["e04_predeclared_rules"][
            "confirmatory_ablation_family"
        ],
        "evidence_class": "derived_qualification_freeze",
    }
    write(OUTPUTS["E04"], e04)

    selected_ids = [c_seq[0], c_cold[0], finalist]
    frozen_candidates = {
        row["config_id"]: row
        for row in manifest["candidates"]
        if row["config_id"] in selected_ids
    }
    e05 = {
        "stage_id": "E05",
        "protocol_id": manifest["protocol_id"],
        "state": "FROZEN",
        "status": "FREEZE_MANIFEST_COMPLETE",
        "ended_utc": now,
        "git_commit": git_commit,
        "qualification_manifest": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "qualification_manifest_sha256": manifest_hash,
        "source_freeze_receipts": {
            stage: {
                "path": str(OUTPUTS[stage].relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest(OUTPUTS[stage]),
            }
            for stage in ("E02", "E03", "E04")
        },
        "C_seq": c_seq[0],
        "C_cold": c_cold[0],
        "MOSAIC_finalist": finalist,
        "frozen_candidates": frozen_candidates,
        "frozen_seeds": [20260924, 20260925, 20260926],
        "data": {
            "fit_visible": manifest["data"]["fit_visible"],
            "controlled_cold_ids": manifest["data"]["controlled_cold_ids"],
            "locked_final_sha256": manifest["data"][
                "locked_final_sha256_recorded_but_access_prohibited"
            ],
        },
        "evaluation": manifest["evaluation"],
        "warm_noninferiority_margin": margin,
        "statistics": e04["statistics"],
        "robustness_family": e04["robustness_family"],
        "confirmatory_ablation_family": e04["confirmatory_ablation_family"],
        "locked_access": "PROHIBITED_UNTIL_THIS_RECEIPT_IS_COMMITTED_AND_PROTOCOL_GUARD_PASSES",
        "warnings": [
            "Qualification metrics are selection evidence, not locked outcomes.",
            "Any scientific mutation after locked access requires a new protocol identity.",
        ],
        "evidence_class": "derived_frozen_protocol_manifest",
    }
    write(OUTPUTS["E05"], e05)
    print(
        json.dumps(
            {
                "C_seq": c_seq[0],
                "C_cold": c_cold[0],
                "MOSAIC_finalist": finalist,
                "warm_noninferiority_margin": margin,
                "E05_sha256": digest(OUTPUTS["E05"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
