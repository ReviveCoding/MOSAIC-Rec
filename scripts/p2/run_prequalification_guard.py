"""Run and persist the clean p2 pre-QUALIFICATION protocol guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.contracts import write_json_new
from mosaic_rec.p2.protocol_guard import prequalification_guard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prequalification_guard(args.repo)
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
