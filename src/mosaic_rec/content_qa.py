"""Check item-to-content mapping without using future engagement aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(dataset: str, interactions: Path, content: Path) -> dict:
    if dataset == "microlens-100k":
        with interactions.open(newline="") as handle:
            items = {int(row["item"]) for row in csv.DictReader(handle)}
        ids = []
        empty = 0
        with content.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, separator, title = line.partition(",")
                if not separator:
                    raise ValueError(f"Malformed title line: {line[:80]}")
                ids.append(int(key.strip()))
                empty += not bool(title.strip())
        columns_excluded = []
    elif dataset == "pixelrec50k":
        with interactions.open(newline="") as handle:
            items = {row["item_id"] for row in csv.DictReader(handle)}
        with content.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            ids = []
            empty = 0
            for row in reader:
                ids.append(row["item_id"])
                empty += not bool((row["title"] or "").strip())
            columns_excluded = [
                col for col in (reader.fieldnames or []) if col.endswith("_number")
            ]
    else:
        raise ValueError(dataset)
    mapped = set(ids)
    result = {
        "stage_id": "A03",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "PILOT",
        "status": "PASS" if items <= mapped and len(ids) == len(mapped) else "FAIL",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "content_file": content.name,
        "content_sha256": sha256(content),
        "interaction_items": len(items),
        "content_rows": len(ids),
        "unique_content_items": len(mapped),
        "interaction_items_missing_content": len(items - mapped),
        "content_items_without_interactions": len(mapped - items),
        "empty_titles": empty,
        "time_unsafe_aggregate_columns_excluded_from_features": columns_excluded,
        "warnings": [
            "Titles/descriptions lack effective-time metadata; historical revision leakage remains possible."
        ],
        "evidence_class": "measured_data_qa",
    }
    if result["status"] != "PASS":
        raise ValueError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["microlens-100k", "pixelrec50k"])
    parser.add_argument("interactions", type=Path)
    parser.add_argument("content", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = inspect(args.dataset, args.interactions, args.content)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
