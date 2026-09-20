"""Stream-qualify official PixelRec image vectors and map PixelRec50K item IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import ijson
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def qualify(source: Path, item_info: Path, output: Path, index: Path) -> dict:
    if not source.is_file() or source.suffix == ".part":
        raise FileNotFoundError("A completed official publisher feature JSON is required")
    with item_info.open(newline="", encoding="utf-8") as handle:
        item_ids = [row["item_id"] for row in csv.DictReader(handle)]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("PixelRec50K item_info contains duplicate item IDs")
    ordered = sorted(item_ids)
    positions = {item: pos for pos, item in enumerate(ordered)}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".npy.tmp")
    vectors = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32",
                                         shape=(len(ordered), 1024))
    seen: set[str] = set()
    mapped = 0
    total = 0
    bad = 0
    with source.open("rb") as handle:
        for item, values in ijson.kvitems(handle, "", use_float=True):
            total += 1
            if item not in positions:
                continue
            if item in seen:
                raise ValueError(f"Duplicate publisher vector for {item}")
            vector = np.asarray(values, dtype=np.float32)
            if vector.shape != (1024,) or not np.isfinite(vector).all():
                bad += 1
                continue
            vectors[positions[item]] = vector
            seen.add(item)
            mapped += 1
    vectors.flush()
    del vectors
    missing = len(ordered) - mapped
    result = {
        "stage_id": "B01", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT", "status": "PASS" if not (missing or bad) else "FAIL",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "pixelrec50k", "feature_source": str(source),
        "feature_source_sha256": sha256(source),
        "publisher_vectors_seen": total, "items_expected": len(ordered),
        "items_mapped": mapped, "items_missing": missing, "bad_vectors": bad,
        "feature_shape": [len(ordered), 1024],
        "warnings": [
            "Publisher-provided frozen image vectors; no local image encoder parity or provenance is claimed.",
            "Item content effective times are unknown; historical content revision leakage cannot be ruled out.",
        ],
        "evidence_class": "measured_feature_qa",
    }
    if result["status"] == "PASS":
        os.replace(tmp, output)
        index.write_text("\n".join(ordered) + "\n", encoding="utf-8")
        result.update(feature_path=str(output), feature_sha256=sha256(output),
                      item_index_path=str(index), item_index_sha256=sha256(index))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("item_info", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = qualify(args.source, args.item_info, args.output, args.index)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
