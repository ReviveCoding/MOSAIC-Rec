"""Single-GPU p2 qualification and advisory lock."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch


def compute_pids() -> list[str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def gpu_activity() -> tuple[int, int]:
    """Return allocated MiB and utilization; WSL may hide compute PIDs."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one physical GPU, observed {len(rows)}")
    memory, utilization = (int(value.strip()) for value in rows[0].split(","))
    return memory, utilization


def display_is_active() -> bool:
    """Return whether the physical GPU is directly driving a display."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=display_active",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1 or rows[0] not in {"Enabled", "Disabled"}:
        raise RuntimeError(f"Unexpected display-active response: {rows}")
    return rows[0] == "Enabled"


def compute_engine_pids() -> list[str]:
    """Return compute-engine PIDs from pmon as a second live-process probe."""
    output = subprocess.check_output(["nvidia-smi", "pmon", "-c", "1"], text=True)
    pids: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 3:
            raise RuntimeError(f"Unexpected nvidia-smi pmon row: {line!r}")
        pid, engine_type = fields[1], fields[2]
        if pid != "-" and "C" in engine_type:
            pids.append(pid)
    return pids


def sustained_idle(
    seconds: int = 30,
    interval: int = 5,
    *,
    maximum_memory_mib: int = 256,
    maximum_utilization_percent: int = 5,
) -> None:
    checks = max(2, seconds // interval + 1)
    for index in range(checks):
        pids = {
            pid
            for pid in [*compute_pids(), *compute_engine_pids()]
            if pid != str(os.getpid())
        }
        if pids:
            raise RuntimeError(
                f"Unrelated CUDA compute PIDs are active: {sorted(pids)}"
            )
        memory, utilization = gpu_activity()
        # Aggregate memory/utilization includes the Windows desktop when the
        # laptop is in dGPU-only mode. In that mode two independent NVIDIA
        # process probes above remain the fail-closed CUDA activity gate.
        if not display_is_active() and (
            memory > maximum_memory_mib or utilization > maximum_utilization_percent
        ):
            raise RuntimeError(
                "GPU is materially active even though WSL exposed no compute PID: "
                f"memory_used_mib={memory}, utilization_percent={utilization}"
            )
        if index + 1 < checks:
            time.sleep(interval)


def cuda_qualification() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited")
    device = torch.device("cuda:0")
    values = torch.arange(4096, device=device, dtype=torch.float32)
    result = float((values.square().mean()).item())
    if not torch.isfinite(torch.tensor(result)):
        raise RuntimeError("CUDA qualification produced a non-finite result")
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "vram_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "tensor_result": result,
    }


@contextmanager
def heavy_gpu_lock(path: Path, model_id: str) -> Iterator[dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"p2 heavy GPU lock is active: {path}") from exc
        sustained_idle()
        qualification = cuda_qualification()
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} protocol=p2 model={model_id}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield qualification
        finally:
            handle.seek(0)
            handle.truncate()
            handle.write(f"released_pid={os.getpid()} model={model_id}\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)
