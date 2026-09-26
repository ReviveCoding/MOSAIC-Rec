"""Apply the frozen p2 QUALIFICATION comparator and router selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import load_json, sha256_file, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ["cseq_bert4rec", "ccold_two_tower", "r2", "r3", "r4"]
    receipts = {}
    for name in names:
        path = args.repo / f"evidence/p2/p2-e02-qualification-{name}.json"
        payload = load_json(path)
        if payload.get("status") != "PASS" or payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"Invalid QUALIFICATION receipt: {name}")
        scored = Path(payload["qualification"]["output"])
        if sha256_file(scored) != payload["qualification"]["output_sha256"]:
            raise RuntimeError(f"QUALIFICATION output hash mismatch: {name}")
        receipts[name] = payload
    cseq = receipts["cseq_bert4rec"]["qualification"]["slices"]
    ccold = receipts["ccold_two_tower"]["qualification"]["slices"]
    rows = []
    for name in ("r2", "r3", "r4"):
        slices = receipts[name]["qualification"]["slices"]
        cold = float(slices["controlled_zero"]["ndcg_at_20"])
        cold_base = float(ccold["controlled_zero"]["ndcg_at_20"])
        cold_term = cold - cold_base if cold_base == 0 else cold / cold_base - 1
        warm_term = (
            float(slices["warm"]["ndcg_at_20"]) / float(cseq["warm"]["ndcg_at_20"])
            - 1
            + 0.10
        )
        rows.append(
            {
                "family": name.upper(),
                "cold_term": cold_term,
                "warm_term": warm_term,
                "selection_score": min(cold_term, warm_term),
                "overall_ndcg_at_20": slices["all"]["ndcg_at_20"],
                "controlled_zero_ndcg_at_20": cold,
                "warm_ndcg_at_20": slices["warm"]["ndcg_at_20"],
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            -row["selection_score"],
            -row["overall_ndcg_at_20"],
            row["family"],
        ),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-E02-E03",
        "status": "PASS_QUALIFICATION",
        "C_seq": "CSEQ_BERT4REC",
        "C_cold": "CCOLD_TWO_TOWER",
        "router_finalist": selected["family"],
        "router_candidates": rows,
        "warm_noninferiority_margin": 0.001320914873267875,
        "qualification_access_count": 1,
        "locked_final_access_count": 0,
        "receipt_sha256": {
            name: sha256_file(
                args.repo / f"evidence/p2/p2-e02-qualification-{name}.json"
            )
            for name in names
        },
        "selection_is_promotion": False,
    }
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
