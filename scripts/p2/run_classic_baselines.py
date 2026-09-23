"""Run exact CPU p2 popularity baselines on open validation queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_rec.p2.classic import run_classic
from mosaic_rec.p2.contracts import verify_p1_immutable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--item-context", type=Path, required=True)
    parser.add_argument("--item-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    before = verify_p1_immutable(args.repo)
    result = run_classic(
        repo=args.repo,
        config_path=args.config,
        fit_path=args.fit,
        query_path=args.queries,
        item_context_path=args.item_context,
        item_ids_path=args.item_ids,
        output_dir=args.output_dir,
        receipt=args.receipt,
    )
    if before != verify_p1_immutable(args.repo):
        raise RuntimeError("p1 changed during p2 classic baseline run")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
