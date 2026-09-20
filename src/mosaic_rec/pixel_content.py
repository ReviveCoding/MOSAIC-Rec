"""Align licensed local PixelRec publisher image and text features by item ID."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualify(feature_dir: Path, metadata_csv: Path, receipt: Path) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    image_ids_path = feature_dir / "publisher_image_item_ids.txt"
    text_ids_path = feature_dir / "local_minilm_item_ids.txt"
    image_ids = image_ids_path.read_text().splitlines()
    text_ids = text_ids_path.read_text().splitlines()
    if image_ids != text_ids or len(image_ids) != 82865 or len(set(image_ids)) != len(image_ids):
        raise ValueError("PixelRec image/text item order is missing, duplicated, or mismatched")
    image = np.load(feature_dir / "publisher_image_features.npy", mmap_mode="r")
    text = np.load(feature_dir / "local_minilm_text.npy", mmap_mode="r")
    if image.shape != (len(image_ids), 1024) or text.shape != (len(image_ids), 384):
        raise ValueError("PixelRec image/text feature dimensions mismatch declared encoders")
    for array in (image, text):
        for start in range(0, len(image_ids), 4096):
            if not np.isfinite(array[start:start + 4096]).all():
                raise ValueError("Nonfinite PixelRec feature detected")
    metadata: dict[str, tuple[str, bool]] = {}
    with metadata_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            item = row["item_id"]
            if item in metadata:
                raise ValueError(f"Duplicate PixelRec metadata item {item}")
            metadata[item] = ((row.get("tag") or "").strip() or "<missing>",
                              bool((row.get("title") or "").strip()))
    if set(metadata) != set(image_ids):
        raise ValueError("PixelRec metadata and feature item ID sets differ")
    tags = sorted({metadata[item][0] for item in image_ids})
    tag_index = {tag: idx for idx, tag in enumerate(tags)}
    tag_ids = np.asarray([tag_index[metadata[item][0]] for item in image_ids], dtype=np.int32)
    text_present = np.asarray([metadata[item][1] for item in image_ids], dtype=np.bool_)
    tag_path = feature_dir / "aligned_tag_ids.npy"
    present_path = feature_dir / "aligned_text_present.npy"
    if tag_path.exists() or present_path.exists() or receipt.exists():
        raise FileExistsError("Aligned PixelRec output exists; preserve it")
    np.save(tag_path, tag_ids)
    np.save(present_path, text_present)
    return {
        "stage_id": "B01", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT", "status": "PASS_ALIGNED_LOCAL_FEATURES",
        "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "pixelrec50k", "model": "CONTENT_ALIGNMENT",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "items": len(image_ids), "image_dim": 1024, "text_dim": 384,
        "tag_categories": len(tags), "empty_titles": int((~text_present).sum()),
        "input_sha256": {
            "publisher_image_item_ids": sha256(image_ids_path),
            "local_text_item_ids": sha256(text_ids_path),
            "publisher_image_features": sha256(feature_dir / "publisher_image_features.npy"),
            "local_text_features": sha256(feature_dir / "local_minilm_text.npy"),
            "item_metadata": sha256(metadata_csv),
        },
        "output_sha256": {"aligned_tag_ids": sha256(tag_path),
                          "aligned_text_present": sha256(present_path)},
        "outputs": {"aligned_tag_ids": str(tag_path),
                    "aligned_text_present": str(present_path)},
        "warnings": [
            "Publisher image and locally extracted text embeddings have different encoder provenance.",
            "Historical content effective times are unknown; no protected result or replication claim follows.",
            "Engagement aggregate columns are excluded from metadata features.",
        ],
        "evidence_class": "measured_content_alignment_qa",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("metadata_csv", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = qualify(args.feature_dir, args.metadata_csv, args.receipt)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
