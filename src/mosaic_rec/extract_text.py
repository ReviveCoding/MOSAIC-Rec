"""Frozen local CUDA text embedding extraction with pinned encoder revision."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def texts_for(dataset: str, path: Path) -> list[str]:
    if dataset == "microlens-100k":
        rows: dict[int, str] = {}
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                item, _, title = line.partition(",")
                rows[int(item)] = title.strip()
        if set(rows) != set(range(1, 19739)):
            raise ValueError("MicroLens title mapping incomplete")
        return [rows[i] for i in range(1, 19739)]
    if dataset == "pixelrec50k":
        rows: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[row["item_id"]] = " ".join(
                    part.strip() for part in (row["title"], row["tag"], row["description"])
                    if part and part.strip()
                )
        # Lexical IDs are persisted separately and must be used for alignment.
        return [rows[item] for item in sorted(rows)]
    raise ValueError(dataset)


def extract(dataset: str, source: Path, output: Path, index: Path,
            cache: Path, lock: Path, batch_size: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for embedding extraction")
    texts = texts_for(dataset, source)
    model_sha = HfApi().model_info(MODEL_ID).sha
    cache.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        lock_handle.write(f"pid={__import__('os').getpid()} model=TEXT_EXTRACTION\n")
        lock_handle.flush()
        model = SentenceTransformer(MODEL_ID, revision=model_sha,
                                    device="cuda", cache_folder=str(cache))
        if str(model.device) != "cuda:0":
            raise RuntimeError(f"Encoder moved off CUDA: {model.device}")
        probe = [text for text in texts if text][:32]
        fp32 = model.encode(probe, batch_size=32, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = model.encode(probe, batch_size=32, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        cosine = np.sum(fp32 * bf16, axis=1)
        parity_min_cosine = float(cosine.min())
        use_bf16 = parity_min_cosine > 0.995
        start = time.perf_counter()
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model.encode(texts, batch_size=batch_size,
                                          normalize_embeddings=True,
                                          convert_to_numpy=True,
                                          show_progress_bar=False)
        else:
            embeddings = model.encode(texts, batch_size=batch_size,
                                      normalize_embeddings=True,
                                      convert_to_numpy=True,
                                      show_progress_bar=False)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        embeddings = np.asarray(embeddings, dtype=np.float32)
        missing = np.asarray([not bool(text) for text in texts])
        embeddings[missing] = 0
        if not np.isfinite(embeddings).all():
            raise ValueError("Non-finite local text embedding")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, embeddings)
        if dataset == "microlens-100k":
            item_ids = [str(i) for i in range(1, 19739)]
        else:
            with source.open(newline="", encoding="utf-8") as handle:
                item_ids = sorted(row["item_id"] for row in csv.DictReader(handle))
        index.write_text("\n".join(item_ids) + "\n")
        peak_vram = torch.cuda.max_memory_allocated()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "stage_id": "B00", "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "DEVELOPMENT", "status": "PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "encoder": MODEL_ID, "encoder_revision": model_sha,
        "encoder_license": "Apache-2.0", "device": torch.cuda.get_device_name(0),
        "batch_size": batch_size, "bf16_used": use_bf16,
        "probe_min_fp32_bf16_cosine": parity_min_cosine,
        "items": len(texts), "missing_text_items": int(missing.sum()),
        "feature_shape": list(embeddings.shape), "seconds": duration,
        "items_per_second": len(texts) / duration,
        "peak_vram_bytes": peak_vram,
        "features_path": str(output), "features_sha256": digest,
        "item_index_path": str(index),
        "warnings": [
            "Encoder weights are frozen; this is local text extraction only.",
            "Source text effective times are unknown; historical revision leakage cannot be ruled out.",
        ],
        "evidence_class": "measured_cuda_feature_extraction",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["microlens-100k", "pixelrec50k"])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("item_index", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    result = extract(args.dataset, args.source, args.output, args.item_index,
                     args.cache, args.lock, args.batch_size)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
