"""Create p2-owned PixelRec feature identities from authorized local sources."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mosaic_rec.p2 import PROTOCOL_ID
from mosaic_rec.p2.contracts import sha256_file, write_json_new


def _copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output, source.open("rb") as input_handle:
        shutil.copyfileobj(input_handle, output, length=16 * 1024 * 1024)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def qualify_pixelrec_features(
    *,
    repo: Path,
    item_source: Path,
    image_source: Path,
    image_index_source: Path,
    text_source: Path,
    text_index_source: Path,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite p2 features: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "image": output_dir / "publisher_image_features.npy",
        "image_index": output_dir / "publisher_image_item_ids.txt",
        "text": output_dir / "authorized_minilm_text_features.npy",
        "text_index": output_dir / "authorized_minilm_item_ids.txt",
    }
    _copy_new(image_source, targets["image"])
    _copy_new(image_index_source, targets["image_index"])
    _copy_new(text_source, targets["text"])
    _copy_new(text_index_source, targets["text_index"])

    image_ids = targets["image_index"].read_text(encoding="utf-8").splitlines()
    text_ids = targets["text_index"].read_text(encoding="utf-8").splitlines()
    if image_ids != text_ids or len(image_ids) != len(set(image_ids)):
        raise RuntimeError("Image/text item indices do not align one-to-one")
    image = np.load(targets["image"], mmap_mode="r")
    text = np.load(targets["text"], mmap_mode="r")
    if image.shape != (len(image_ids), 1024):
        raise RuntimeError(f"Unexpected publisher image shape: {image.shape}")
    if text.shape != (len(text_ids), 384):
        raise RuntimeError(f"Unexpected text shape: {text.shape}")
    image_finite = bool(np.isfinite(image).all())
    text_finite = bool(np.isfinite(text).all())
    if not image_finite or not text_finite:
        raise RuntimeError("Non-finite p2 content feature")

    rows: dict[str, dict[str, str]] = {}
    with item_source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["item_id"]] = row
    if sorted(rows) != image_ids:
        raise RuntimeError("Item metadata and feature indices differ")
    tags = sorted(
        {rows[item]["tag"].strip() for item in image_ids if rows[item]["tag"].strip()}
    )
    tag_to_id = {tag: index for index, tag in enumerate(tags)}
    tag_ids = np.asarray(
        [tag_to_id.get(rows[item]["tag"].strip(), -1) for item in image_ids],
        dtype=np.int32,
    )
    text_present = np.asarray(
        [
            any(rows[item][field].strip() for field in ("title", "tag", "description"))
            for item in image_ids
        ],
        dtype=np.bool_,
    )
    image_present = np.ones(len(image_ids), dtype=np.bool_)
    np.save(output_dir / "tag_ids.npy", tag_ids)
    np.save(output_dir / "text_present.npy", text_present)
    np.save(output_dir / "image_present.npy", image_present)
    (output_dir / "tag_vocabulary.json").write_text(
        json.dumps(tags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    inputs = {
        "item_source": sha256_file(item_source),
        "image_source": sha256_file(image_source),
        "image_index_source": sha256_file(image_index_source),
        "text_source": sha256_file(text_source),
        "text_index_source": sha256_file(text_index_source),
        "content_amendment": sha256_file(
            repo / "configs/p2/amendment_001_content_source.json"
        ),
    }
    outputs = {
        path.name: {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    common = {
        "protocol_id": PROTOCOL_ID,
        "git_commit": _git_head(repo),
        "dataset": "pixelrec50k",
        "inputs": inputs,
        "outputs": outputs,
    }
    receipts = {
        "P2-B00": {
            **common,
            "stage_id": "P2-B00",
            "status": "PASS_REUSED_AUTHORIZED_BYTES_NEW_P2_IDENTITY",
            "items": len(image_ids),
            "image_shape": list(image.shape),
            "text_shape": list(text.shape),
            "image_finite": image_finite,
            "text_finite": text_finite,
            "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "text_encoder_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "image_source_type": "official publisher extracted feature",
            "text_source_type": "authorized local embedding of official title/tag/description",
            "cuda_extraction_rerun": False,
            "evidence_class": "p2_feature_identity_materialization",
        },
        "P2-B01": {
            **common,
            "stage_id": "P2-B01",
            "status": "PASS",
            "items": len(image_ids),
            "aligned_indices": True,
            "unique_indices": len(set(image_ids)),
            "tag_categories": len(tags),
            "text_missing": int((~text_present).sum()),
            "image_missing": int((~image_present).sum()),
            "excluded_engagement_fields": [
                "view_number",
                "comment_number",
                "thumbup_number",
                "share_number",
                "coin_number",
                "favorite_number",
                "barrage_number",
            ],
            "warnings": [
                "Publisher image and local text embeddings have different encoder provenance.",
                "Historical content effective times are unavailable.",
                "Feature bytes are local noncommercial research inputs and are not redistributed.",
            ],
            "evidence_class": "p2_measured_feature_alignment_qa",
        },
    }
    for stage, receipt in receipts.items():
        write_json_new(repo / f"evidence/p2/{stage.lower()}_pixelrec.json", receipt)
    return receipts
