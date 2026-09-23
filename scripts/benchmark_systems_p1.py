"""Measured CUDA/system evidence after frozen scientific identities."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from fastapi.testclient import TestClient

from mosaic_rec.serving import create_app

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/serving_development_p1.json"
OUTPUT = ROOT / "evidence/j00_j04_systems_p1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Systems receipt exists; preserve it")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for neural systems benchmark")
    serving_manifest = json.loads(CONFIG.read_text())
    fit = pq.read_table(
        Path(serving_manifest["split_dir"]) / "FIT_VISIBLE.parquet",
        columns=["user_id"],
    )
    users = sorted({int(user) for user in fit["user_id"].to_pylist()})[:120]
    compile_receipt = json.loads((ROOT / "evidence/j00_compile_p1.json").read_text())
    compile_result = compile_receipt["measurement"]

    app = create_app(CONFIG)
    api_started = time.perf_counter()
    with TestClient(app) as client:
        api_startup_seconds = time.perf_counter() - api_started
        for user in users[:10]:
            response = client.post("/recommend", json={"user_id": user, "k": 20})
            if response.status_code != 200:
                raise RuntimeError("API warmup failed")
        latencies = []
        errors = 0
        started = time.perf_counter()
        for user in users[:100]:
            call = time.perf_counter()
            response = client.post("/recommend", json={"user_id": user, "k": 20})
            latencies.append((time.perf_counter() - call) * 1000)
            errors += int(response.status_code != 200)
        elapsed = time.perf_counter() - started
        diagnostic_latencies = []
        for user in users[:30]:
            call = time.perf_counter()
            response = client.post("/diagnostics", json={"user_id": user, "k": 20})
            diagnostic_latencies.append((time.perf_counter() - call) * 1000)
            errors += int(response.status_code != 200)

    locked = {
        name: json.loads((ROOT / "evidence" / filename).read_text())
        for name, filename in {
            "C_seq": "f00_locked_p1_b7_bert4rec_t3.json",
            "C_cold": "f00_locked_p1_b4_two_tower_t3.json",
            "MOSAIC": "f00_locked_p1_m1_t2.json",
        }.items()
    }
    training = json.loads(
        (ROOT / "evidence/e01_seedqual_p1_m1_t2_seed20260924.json").read_text()
    )
    raw_training = json.loads(Path(training["raw_receipt"]).read_text())
    microlens_text = json.loads(
        (ROOT / "evidence/microlens_local_text_extraction.json").read_text()
    )
    pixel_text = json.loads(
        (ROOT / "evidence/pixelrec_local_text_extraction.json").read_text()
    )
    result = {
        "stage_id": "J00_J04",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "ANALYSIS",
        "status": "SYSTEMS_STUDY_PARTIAL_PASS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": torch.cuda.get_device_name(0),
        "feature_extraction": {
            "microlens_text_items_per_second": microlens_text["items_per_second"],
            "microlens_peak_vram_bytes": microlens_text["peak_vram_bytes"],
            "pixelrec_text_items_per_second": pixel_text["items_per_second"],
            "pixelrec_peak_vram_bytes": pixel_text["peak_vram_bytes"],
        },
        "training": {
            "model": "M1_T2_seed20260924",
            "fit_samples": raw_training["fit_samples"],
            "epochs": raw_training["epochs"],
            "train_seconds": training["train_seconds"],
            "sample_epochs_per_second": (
                raw_training["fit_samples"]
                * raw_training["epochs"]
                / training["train_seconds"]
            ),
            "peak_vram_bytes": training["peak_vram_bytes"],
        },
        "exact_full_catalog": {
            name: {
                "users": receipt["per_user_rows"],
                "seconds": receipt["score_seconds"],
                "users_per_second": receipt["per_user_rows"] / receipt["score_seconds"],
                "peak_vram_bytes": receipt["peak_vram_bytes"],
            }
            for name, receipt in locked.items()
        },
        "torch_compile": compile_result,
        "api": {
            "manifest": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": digest(CONFIG),
            "evidence_class": "development_demonstration_only",
            "backend_startup_seconds": api_startup_seconds,
            "api_startup_seconds": api_startup_seconds,
            "recommend_requests": len(latencies),
            "recommend_p50_ms": percentile(latencies, 0.50),
            "recommend_p95_ms": percentile(latencies, 0.95),
            "recommend_mean_ms": statistics.mean(latencies),
            "sequential_throughput_requests_per_second": len(latencies) / elapsed,
            "diagnostic_requests": len(diagnostic_latencies),
            "diagnostic_p50_ms": percentile(diagnostic_latencies, 0.50),
            "diagnostic_p95_ms": percentile(diagnostic_latencies, 0.95),
            "errors": errors,
        },
        "ann": {
            "status": "HOLD_NOT_EXECUTED",
            "reason": "No ANN index family or accuracy/latency operating point was frozen; exact ranking remains the scientific path.",
        },
        "warnings": [
            "API evidence uses the declared development demo checkpoint, not the frozen three-seed scientific ensemble.",
            "Single-process sequential request throughput is a local demonstration, not production capacity.",
            "Timing from E01 and locked scoring is factual workload evidence, not a cross-model hardware-normalized comparison.",
        ],
        "evidence_class": "measured_local_systems_partial",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
