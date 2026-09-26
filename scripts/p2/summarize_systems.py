"""Summarize already measured p2 system evidence without new model execution."""

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
    seeds = []
    for seed in (20261001, 20261002, 20261003):
        receipt = load_json(args.repo / f"evidence/p2/p2-e01-e01-r4-seed{seed}.json")
        seeds.append(
            {
                "seed": seed,
                "train_seconds": receipt["train_seconds"],
                "training_examples": receipt["training_examples"],
                "examples_per_second": receipt["training_examples"]
                / receipt["train_seconds"],
                "peak_vram_bytes": receipt["peak_vram_bytes"],
            }
        )
    scoring = {}
    for stem, name in (
        ("cseq_bert4rec", "C_seq"),
        ("ccold_two_tower", "C_cold"),
        ("r4", "R4"),
    ):
        receipt = load_json(args.repo / f"evidence/p2/p2-f00-locked-{stem}.json")
        path = Path(receipt["locked_final"]["output"])
        scoring[name] = {
            "seconds": receipt["locked_final"]["seconds"],
            "users_per_second": receipt["locked_final"]["users_per_second"],
            "topk_artifact_bytes": path.stat().st_size,
            "topk_artifact_sha256": sha256_file(path),
        }
    result = {
        "protocol_id": PROTOCOL_ID,
        "stage_id": "P2-J00-J05",
        "status": "PARTIAL_PASS",
        "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
        "physical_gpu_count": 1,
        "r4_training": seeds,
        "locked_exact_scoring": scoring,
        "holds": {
            "feature_loading_throughput": "HOLD_NOT_MEASURED",
            "local_api_latency_and_requests_per_second": "HOLD_NO_FROZEN_SERVICE_BENCHMARK",
            "torch_compile": "HOLD_NOT_USED_NO_BREAK_EVEN_STUDY",
            "multi_gpu": "PROHIBITED_NOT_MEASURED",
            "production_readiness": "PROHIBITED_UNSUPPORTED",
        },
        "evidence_class": "p2_measured_system_partial",
    }
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
