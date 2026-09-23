"""Measure torch.compile in an isolated process for the serving workload."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mosaic_rec.serving import ServingBackend

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/serving_development_p1.json"
OUTPUT = ROOT / "evidence/j00_compile_p1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Compile receipt exists; preserve it")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for compile benchmark")
    backend = ServingBackend(CONFIG)
    try:
        users = sorted(backend.histories)[:64]
        recent = [backend.histories[user][-20:] for user in users]
        sequence = torch.tensor(
            [[x[0] for x in row] + [0] * (20 - len(row)) for row in recent],
            device=backend.device,
        )
        support = torch.tensor(
            [[x[1] for x in row] + [0] * (20 - len(row)) for row in recent],
            device=backend.device,
        )
        lengths = torch.tensor([len(row) for row in recent], device=backend.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                eager_output = backend.model.user_state(sequence, support, lengths)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                eager_output = backend.model.user_state(sequence, support, lengths)
            torch.cuda.synchronize()
            eager_seconds = (time.perf_counter() - started) / 20
        try:
            compiled = torch.compile(backend.model.user_state)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                compiled_output = compiled(sequence, support, lengths)
            torch.cuda.synchronize()
            cold_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(20):
                    compiled_output = compiled(sequence, support, lengths)
            torch.cuda.synchronize()
            compiled_seconds = (time.perf_counter() - started) / 20
            parity = float(
                (eager_output.float() - compiled_output.float()).abs().max().item()
            )
            savings = eager_seconds - compiled_seconds
            measurement: dict[str, object] = {
                "status": "PASS",
                "batch_size": len(recent),
                "compile_cold_seconds": cold_seconds,
                "eager_steady_seconds_per_batch": eager_seconds,
                "compiled_steady_seconds_per_batch": compiled_seconds,
                "speedup": eager_seconds / compiled_seconds,
                "max_abs_parity": parity,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "break_even_batches": cold_seconds / savings if savings > 0 else None,
                "adopt": savings > 0 and parity <= 1e-5,
            }
        except RuntimeError as exc:
            measurement = {
                "status": "FAIL_PRESERVED",
                "exception": repr(exc),
                "adopt": False,
            }
    finally:
        backend.close()
    result = {
        "stage_id": "J00_COMPILE",
        "protocol_id": "mosaic-rec-2026-09-20-p1",
        "state": "ANALYSIS",
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "device": torch.cuda.get_device_name(0),
        "measurement": measurement,
        "evidence_class": "measured_local_systems",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
