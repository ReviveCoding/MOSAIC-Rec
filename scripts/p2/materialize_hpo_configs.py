"""Materialize open-data trial configs from the frozen p2 HPO manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import verify_p1_immutable, write_json_new
from mosaic_rec.p2.hpo_configs import (
    materialize_open_hpo_configs,
    write_materialized_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    before = verify_p1_immutable(args.repo)
    hpo_path = args.repo / "configs/p2/hpo.json"
    configs = materialize_open_hpo_configs(
        hpo=json.loads(hpo_path.read_text()),
        development=json.loads(
            (args.repo / "configs/p2/development_pilots_attempt2.json").read_text()
        ),
        additional=json.loads(
            (args.repo / "configs/p2/additional_baseline_pilots.json").read_text()
        ),
        legacy=json.loads((args.repo / "configs/p2/legacy_m1_pilot.json").read_text()),
    )
    result = write_materialized_configs(
        output_dir=args.output_dir, configs=configs, hpo_path=hpo_path
    )
    write_json_new(args.manifest, result)
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed while materializing p2 HPO configs")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
