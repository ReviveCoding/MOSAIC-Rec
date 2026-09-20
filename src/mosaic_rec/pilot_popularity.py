"""MicroLens validation-only exact popularity pilot; no locked outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path


def evaluate(source: Path, fit_cutoff: int, validation_cutoff: int) -> dict:
    events: list[tuple[int, int, int]] = []
    first_seen: dict[int, int] = {}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            user, item, stamp = (
                int(row["user"]),
                int(row["item"]),
                int(row["timestamp"]),
            )
            events.append((stamp, user, item))
            first_seen[item] = min(first_seen.get(item, stamp), stamp)
    events.sort()
    fit_counts: Counter[int] = Counter()
    fit_seen: dict[int, set[int]] = defaultdict(set)
    validation_first: dict[int, tuple[int, int]] = {}
    for stamp, user, item in events:
        if stamp <= fit_cutoff:
            fit_counts[item] += 1
            fit_seen[user].add(item)
        elif stamp <= validation_cutoff and user not in validation_first:
            validation_first[user] = (stamp, item)
    # A descending score sort with deterministic ID ties is an exact full-catalog
    # ordering for the frozen FIT popularity vector.
    ranking = sorted(first_seen, key=lambda item: (-fit_counts[item], item))
    hits = reciprocal = discounted = 0.0
    evaluated = 0
    no_fit = unavailable = repeat_target = 0
    for user, (stamp, target) in validation_first.items():
        if user not in fit_seen:
            no_fit += 1
            continue
        if first_seen[target] > stamp:
            unavailable += 1
            continue
        if target in fit_seen[user]:
            repeat_target += 1
            continue
        seen = fit_seen[user]
        top20: list[int] = []
        for item in ranking:
            if first_seen[item] <= stamp and item not in seen:
                top20.append(item)
                if len(top20) == 20:
                    break
        evaluated += 1
        if target in top20:
            rank = top20.index(target) + 1
            hits += 1
            reciprocal += 1 / rank
            discounted += 1 / log2(rank + 1)
    if not evaluated:
        raise ValueError("No eligible validation examples")
    return {
        "stage_id": "C00",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "PILOT",
        "status": "PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "model": "B0_GLOBAL_FIT_POPULARITY",
        "role": "VALIDATION",
        "fit_cutoff": fit_cutoff,
        "validation_cutoff": validation_cutoff,
        "evaluation_population": "first validation event per FIT-seen user with unseen target",
        "ranking": "exact all available items, FIT-only popularity, item-ID tie break",
        "candidate_availability": "first observed timestamp <= event timestamp",
        "evaluated_users": evaluated,
        "excluded_no_fit_user": no_fit,
        "excluded_target_already_seen": repeat_target,
        "excluded_unavailable_target": unavailable,
        "ndcg_at_20": discounted / evaluated,
        "recall_at_20": hits / evaluated,
        "mrr_at_20": reciprocal / evaluated,
        "hit_rate_at_20": hits / evaluated,
        "warnings": [
            "Pilot validation only; no controlled cold cohort, HPO, or locked inference.",
            "Unobserved items are not asserted to be true negatives.",
            "First observed time is a visibility proxy, not launch time.",
        ],
        "evidence_class": "measured_pilot_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--fit-cutoff", type=int, required=True)
    parser.add_argument("--validation-cutoff", type=int, required=True)
    args = parser.parse_args()
    result = evaluate(args.source, args.fit_cutoff, args.validation_cutoff)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
