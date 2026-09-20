"""Prepare FIT-safe MicroLens publisher vectors for isolated MMRec reference pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SOURCES = {
    "vision": ("MicroLens-100k_image_features_CLIPRN50.npy", "image"),
    "text": ("MicroLens-100k_title_en_text_features_BgeM3.npy", "text"),
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def prepare(source_dir: Path, feature_receipt: Path, destination: Path) -> dict:
    expected = json.loads(feature_receipt.read_text())["features"]
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for label, (filename, receipt_key) in SOURCES.items():
        source = source_dir / filename
        source_hash = sha256(source)
        if source_hash != expected[receipt_key]["npy_sha256"]:
            raise ValueError(f"Publisher {label} source hash mismatch")
        content = np.load(source)
        if content.shape != (19738, 1024) or not np.isfinite(content).all():
            raise ValueError(f"Invalid publisher {label} matrix")
        padded = np.zeros((19739, 1024), dtype=np.float32)
        padded[1:] = content
        path = destination / f"{label}.npy"
        np.save(path, padded)
        outputs[label] = {
            "path": str(path),
            "sha256": sha256(path),
            "shape": list(padded.shape),
            "source_sha256": source_hash,
        }
    return {
        "stage_id": "C04",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT",
        "status": "REFERENCE_INPUT_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "reference": "MMRec",
        "outputs": outputs,
        "warnings": [
            "Adds only the index-zero padding row required by MMRec; item IDs 1..19738 retain publisher alignment.",
            "Reference input preparation is not a model score or parity result.",
        ],
        "evidence_class": "derived_reference_input",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("feature_receipt", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = prepare(args.source_dir, args.feature_receipt, args.destination)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
