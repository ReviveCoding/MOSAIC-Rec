"""Materialize the preregistered p2 score-alignment candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import sha256_file, verify_p1_immutable, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.repo / "configs/p2/alignment_development.json"
    implementation_path = args.repo / "configs/p2/alignment_implementation_001.json"
    router_path = args.repo / "configs/p2/router_pilots.json"
    source = json.loads(source_path.read_text())
    router = json.loads(router_path.read_text())
    before = verify_p1_immutable(args.repo)
    output_dir = args.repo / "configs/p2/alignment_trials"
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for candidate in source["candidates"]:
        alignment_id = candidate["alignment_id"]
        path = output_dir / f"{alignment_id.lower()}.json"
        trial = {
            "protocol_id": source["protocol_id"],
            "stage_id": source["stage_id"],
            "router": {
                "model_id": f"R1_{alignment_id}",
                "definition": router["router"]["definition"],
                "alignment": candidate["method"],
                "standardization_population": router["router"][
                    "standardization_population"
                ],
                "zero_variance_scale": router["router"]["zero_variance_scale"],
            },
            "experts": router["experts"],
            "scoring": router["scoring"],
            "alignment_id": alignment_id,
            "source_alignment_sha256": sha256_file(source_path),
            "implementation_sha256": sha256_file(implementation_path),
            "evidence_class": "p2_measured_development_score_alignment",
            "protected_roles_opened": [],
        }
        if "temperatures" in candidate:
            trial["router"]["temperatures"] = candidate["temperatures"]
        write_json_new(path, trial)
        trials.append(
            {
                "alignment_id": alignment_id,
                "path": str(path.relative_to(args.repo)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "protocol_id": source["protocol_id"],
        "stage_id": "P2-B03-ALIGNMENT-MATERIALIZATION",
        "status": "PASS",
        "source_alignment_sha256": sha256_file(source_path),
        "implementation_sha256": sha256_file(implementation_path),
        "trial_count": len(trials),
        "trials": trials,
        "protected_roles_opened": [],
    }
    write_json_new(args.repo / "configs/p2/alignment_trials_manifest.json", manifest)
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 alignment materialization")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
