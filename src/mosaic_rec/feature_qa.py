"""Publisher-provided MicroLens feature integrity and item-row alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(feature_dir: Path) -> dict:
    modalities = {
        "image": "MicroLens-100k_image_features_CLIPRN50",
        "text": "MicroLens-100k_title_en_text_features_BgeM3",
        "video_secondary": "MicroLens-100k_video_features_VideoMAE",
    }
    results: dict[str, dict] = {}
    for modality, stem in modalities.items():
        array_path = feature_dir / f"{stem}.npy"
        json_path = feature_dir / f"{stem}.json"
        vectors = np.load(array_path, mmap_mode="r")
        with json_path.open() as handle:
            mapping = json.load(handle)
        keys = sorted(map(int, mapping))
        if keys != list(range(1, 19739)):
            raise ValueError(f"{modality}: key set mismatch")
        if vectors.shape[0] != len(keys):
            raise ValueError(f"{modality}: vector rows != keys")
        # Check complete ID-to-row mapping; a shape-only check misses permutations.
        max_abs_difference = 0.0
        for item_id in keys:
            reference = np.asarray(mapping[str(item_id)], dtype=np.float32)
            difference = float(np.max(np.abs(vectors[item_id - 1] - reference)))
            max_abs_difference = max(max_abs_difference, difference)
        norms = np.linalg.norm(np.asarray(vectors), axis=1)
        results[modality] = {
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "npy_sha256": sha256(array_path),
            "json_sha256": sha256(json_path),
            "finite": bool(np.isfinite(vectors).all()),
            "zero_norm_rows": int(np.count_nonzero(norms == 0)),
            "min_norm": float(norms.min()),
            "median_norm": float(np.median(norms)),
            "max_norm": float(norms.max()),
            "max_abs_json_npy_difference": max_abs_difference,
        }
    return {
        "stage_id": "B01",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "PILOT",
        "status": "PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "microlens-100k",
        "features": results,
        "warnings": [
            "Publisher-provided frozen embeddings; no local encoder extraction or encoder-version parity claimed.",
            "Verify source encoder training provenance before confirmatory use.",
        ],
        "evidence_class": "measured_feature_qa",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = inspect(args.feature_dir)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
