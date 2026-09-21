"""Summarize prespecified development seeds without selecting a best seed."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260920, 20260921, 20260922)
VARIANTS = (1, 4, 6, 7)
SLICES = ("all", "controlled_cold", "warm_fit")


def receipt_path(variant: int, seed: int) -> Path:
    suffix = "pilot" if seed == 20260920 else f"seed{seed}_development"
    return ROOT / "evidence" / f"microlens_m{variant}_{suffix}.json"


def main() -> None:
    output: dict[str, object] = {
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "evidence_class": "derived_development_seed_stability",
        "seeds": list(SEEDS),
        "variants": {},
    }
    variants: dict[str, object] = {}
    for variant in VARIANTS:
        rows = []
        for seed in SEEDS:
            path = receipt_path(variant, seed)
            receipt = json.loads(path.read_text())
            if (
                receipt["protocol_id"] != output["protocol_id"]
                or receipt["seed"] != seed
            ):
                raise ValueError(f"Receipt identity mismatch: {path}")
            if receipt["status"] != "PILOT_PASS":
                raise ValueError(f"Seed run did not pass: {path}")
            rows.append(receipt)
        slices = {}
        for name in SLICES:
            values = [
                float(row["validation_slices"][name]["ndcg_at_20"]) for row in rows
            ]
            slices[name] = {
                "ndcg_at_20_by_seed": dict(zip(map(str, SEEDS), values, strict=True)),
                "mean_ndcg_at_20": statistics.mean(values),
                "sample_sd_ndcg_at_20": statistics.stdev(values),
                "min_ndcg_at_20": min(values),
                "max_ndcg_at_20": max(values),
            }
        variants[f"M{variant}"] = {
            "receipts": [
                str(receipt_path(variant, seed).relative_to(ROOT)) for seed in SEEDS
            ],
            "slices": slices,
        }
    output["variants"] = variants
    destination = ROOT / "evidence" / "microlens_mosaic_seed_stability_p1.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
