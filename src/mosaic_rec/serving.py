"""Local FastAPI demonstration for an explicitly identified MOSAIC checkpoint."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mosaic_rec.pilot_mosaic import MosaicSequence, load_content


BRANCHES = ("collaborative", "text", "image", "metadata")


class RecommendRequest(BaseModel):
    user_id: int = Field(ge=0)
    k: int = Field(default=20, ge=1, le=100)


class ServingBackend:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest = json.loads(manifest_path.read_text())
        m = self.manifest
        if m["protocol_id"] != "mosaic-rec-2026-09-20-p1":
            raise ValueError("Unknown serving protocol")
        if m["state"] != "DEVELOPMENT" or m["evidence_class"] != "demonstration_only":
            raise ValueError("This adapter only accepts a declared development demo")
        checkpoint = Path(m["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != m["checkpoint_sha256"]:
            raise ValueError("Serving checkpoint hash mismatch")
        current_processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if current_processes:
            raise RuntimeError(f"GPU occupied before serving startup: {current_processes}")
        lock = Path("/home/bjw-0/mosaic-rec-work/locks/gpu.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = lock.open("a+")
        try:
            fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_handle.seek(0)
            self._lock_handle.truncate()
            self._lock_handle.write(f"pid={os.getpid()} model={m['model']}_SERVING_DEMO\n")
            self._lock_handle.flush()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for neural serving demonstration")
            self.device = torch.device("cuda:0")
            content = load_content(Path(m["feature_dir"]), Path(m["tag_file"]),
                                   Path(m["title_file"]), self.device)
            checkpoint_data = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if checkpoint_data["variant"] != m["variant"]:
                raise ValueError("Manifest/checkpoint model variant mismatch")
            self.model = MosaicSequence(m["variant"], *content).to(self.device)
            self.model.load_state_dict(checkpoint_data["model"])
            self.model.eval()
            self.histories: dict[int, list[tuple[int, int]]] = defaultdict(list)
            self.seen: dict[int, set[int]] = defaultdict(set)
            counts: dict[int, int] = defaultdict(int)
            fit = pq.read_table(Path(m["split_dir"]) / "FIT_VISIBLE.parquet",
                                columns=["timestamp", "user_id", "item_id"])
            events = sorted(zip(fit["timestamp"].to_pylist(),
                                fit["user_id"].to_pylist(), fit["item_id"].to_pylist()))
            for _, user, item in events:
                user, item = int(user), int(item)
                self.histories[user].append((item, counts[item]))
                self.seen[user].add(item)
                counts[item] += 1
            first_seen = [2**63 - 1] * 19739
            with Path(m["source_csv"]).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    item, stamp = int(row["item"]), int(row["timestamp"])
                    first_seen[item] = min(first_seen[item], stamp)
            self.available = torch.tensor(
                [first_seen[item] <= m["fit_end"] for item in range(1, 19739)],
                device=self.device,
            )
            self.item_ids = torch.arange(1, 19739, device=self.device)
            self.support = torch.tensor([counts[i] for i in range(1, 19739)],
                                        device=self.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                catalog, weights = self.model.fuse(self.item_ids, self.support)
            self.catalog = catalog.float()
            self.weights = weights.float()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_lock_handle", None) is not None:
            fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _state(self, user_id: int) -> torch.Tensor:
        history = self.histories.get(user_id)
        if not history:
            raise KeyError(user_id)
        recent = history[-20:]
        seq = torch.tensor([[x[0] for x in recent] + [0] * (20 - len(recent))],
                           device=self.device)
        support = torch.tensor([[x[1] for x in recent] + [0] * (20 - len(recent))],
                               device=self.device)
        length = torch.tensor([len(recent)], device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.user_state(seq, support, length).float()[0]

    def recommend(self, user_id: int, k: int, diagnostics: bool = False) -> dict[str, Any]:
        state = self._state(user_id)
        with torch.inference_mode():
            scores = (self.catalog @ state).to(torch.float64)
            scores -= torch.arange(19738, device=self.device, dtype=torch.float64) * 1e-12
            scores.masked_fill_(~self.available, -torch.inf)
            scores[torch.tensor(sorted(self.seen[user_id]), device=self.device) - 1] = -torch.inf
            values, indices = scores.topk(k)
            item_ids = (indices + 1).tolist()
            results = [{"item_id": item, "score": float(value)}
                       for item, value in zip(item_ids, values.tolist())]
            if diagnostics:
                for result in results:
                    item = result["item_id"]
                    idx = item - 1
                    if self.model.variant == 0:
                        coefficients = [
                            float(self.support[idx].item() > 0),
                            float(self.model.text_present[idx].item()),
                            1.0,
                            1.0,
                        ]
                        # M0 is a sum, not a learned gate; its old diagnostic
                        # softmax did not control fusion and is never reported.
                        result["gate_weights"] = None
                    else:
                        coefficients = self.weights[idx].tolist()
                        result["gate_weights"] = dict(zip(BRANCHES, coefficients))
                    branch_vectors = (
                        self.model.item(self.item_ids[idx]),
                        self.model.text_projection(self.model.text[idx]),
                        self.model.image_projection(self.model.image[idx]),
                        self.model.tag_embedding(self.model.tags[idx]),
                    )
                    result["branch_score_contributions"] = {
                        name: float(torch.dot(state, vector.float()) * coefficient)
                        for name, vector, coefficient in zip(BRANCHES, branch_vectors, coefficients)
                    }
        return {
            "protocol_id": self.manifest["protocol_id"],
            "model": self.manifest["model"],
            "evidence_class": "demonstration_only",
            "snapshot": "FIT_end",
            "user_id": user_id,
            "recommendations": results,
            "warning": self.manifest["warning"],
        }


def create_app(manifest_path: Path | None = None) -> FastAPI:
    backend: ServingBackend | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal backend
        if manifest_path is not None:
            backend = ServingBackend(manifest_path)
        try:
            yield
        finally:
            if backend is not None:
                backend.close()
                backend = None

    app = FastAPI(title="MOSAIC-Rec development demonstration", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ready" if backend is not None else "unconfigured",
            "evidence_class": "demonstration_only",
            "model": backend.manifest["model"] if backend is not None else None,
            "gpu": torch.cuda.get_device_name(0) if backend is not None else None,
        }

    def dispatch(request: RecommendRequest, diagnostics: bool) -> dict[str, Any]:
        if backend is None:
            raise HTTPException(status_code=503, detail="No verified development checkpoint loaded")
        try:
            return backend.recommend(request.user_id, request.k, diagnostics=diagnostics)
        except KeyError:
            raise HTTPException(status_code=404, detail="User has no FIT-visible history") from None

    @app.post("/recommend")
    def recommend(request: RecommendRequest) -> dict[str, Any]:
        return dispatch(request, diagnostics=False)

    @app.post("/diagnostics")
    def diagnostics(request: RecommendRequest) -> dict[str, Any]:
        return dispatch(request, diagnostics=True)

    return app


app = create_app(Path(os.environ["MOSAIC_SERVING_MANIFEST"])
                 if "MOSAIC_SERVING_MANIFEST" in os.environ else None)
