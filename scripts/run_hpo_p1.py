"""Run one predeclared neural HPO model family member at a time on CUDA."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/home/bjw-0/mosaic-rec-work")
PY = SCRATCH / ".venv/bin/python"
SPLIT = SCRATCH / "data/microlens-100k/split_p1"
SOURCE = SCRATCH / "data/microlens-100k/pairs.csv"
FEATURES = SCRATCH / "features/microlens-100k"
TAGS = SCRATCH / "data/microlens-100k/tags_to_summary.csv"
TITLES = SCRATCH / "data/microlens-100k/title_en.csv"
LOCK = SCRATCH / "locks/gpu.lock"
POLICY_PATH = ROOT / "configs/hpo_development_p1.json"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_available() -> None:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    )
    if output.strip():
        raise RuntimeError(f"GPU compute process already active: {output.strip()}")
    with LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("MOSAIC heavy GPU lock is owned") from exc
        fcntl.flock(handle, fcntl.LOCK_UN)


def reference_features(trial_name: str) -> Path:
    destination = SCRATCH / "references" / f"hpo_p1_{trial_name}"
    source = SCRATCH / "references/mmrec_micro_p1"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("vision.npy", "text.npy"):
        link = destination / filename
        target = source / filename
        if link.exists() and link.resolve() != target.resolve():
            raise ValueError(f"Reference input identity changed: {link}")
        if not link.exists():
            link.symlink_to(target)
    return destination


def command(
    model: str, receipt: Path, checkpoint: Path, epochs: int, seed: int, trial_name: str
) -> list[str]:
    common = [
        "--checkpoint",
        str(checkpoint),
        "--lock",
        str(LOCK),
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
    ]
    if model == "B2_BPR_MF":
        return [
            str(PY),
            "-m",
            "mosaic_rec.pilot_bpr_split",
            str(SPLIT),
            str(SOURCE),
            str(receipt),
            *common,
        ]
    if model == "B3_LIGHTGCN":
        return [
            str(PY),
            "-m",
            "mosaic_rec.pilot_lightgcn",
            str(SPLIT),
            str(SOURCE),
            str(receipt),
            *common,
        ]
    if model in {"B4_TWO_TOWER", "B8_VBPR"}:
        mode = "two_tower_content" if model == "B4_TWO_TOWER" else "vbpr"
        return [
            str(PY),
            "-m",
            "mosaic_rec.pilot_content",
            mode,
            str(SPLIT),
            str(SOURCE),
            str(FEATURES),
            str(TAGS),
            str(receipt),
            *common,
        ]
    if model in {"B5_GRU4REC", "B6_SASREC"}:
        architecture = "gru4rec" if model == "B5_GRU4REC" else "sasrec"
        return [
            str(PY),
            "-m",
            "mosaic_rec.pilot_sasrec",
            str(SPLIT),
            str(SOURCE),
            str(receipt),
            "--architecture",
            architecture,
            "--manifest",
            str(ROOT / "configs/sequence_coldguard_pilot_p1.json"),
            *common,
        ]
    if model == "B7_BERT4REC":
        return [
            str(PY),
            "-m",
            "mosaic_rec.pilot_bert4rec",
            str(SPLIT),
            str(SOURCE),
            str(receipt),
            "--config",
            str(ROOT / "configs/bert4rec_common_pilot_p1.json"),
            *common,
        ]
    if model in {"B9_BM3", "B10_FREEDOM"}:
        module = (
            "mosaic_rec.pilot_mmrec_bm3"
            if model == "B9_BM3"
            else "mosaic_rec.pilot_mmrec_freedom"
        )
        feature_dir = reference_features(trial_name)
        extra = (
            [] if model == "B9_BM3" else ["--attempt", str(100 + int(trial_name[-1]))]
        )
        return [
            str(PY),
            "-m",
            module,
            str(SCRATCH / "references/MMRec"),
            str(feature_dir),
            str(SPLIT),
            str(SOURCE),
            str(receipt),
            *common,
            *extra,
        ]
    if model in {"M1", "M4", "M6", "M7"}:
        return [
            str(PY),
            "-m",
            "mosaic_rec.pilot_mosaic",
            model[1:],
            str(SPLIT),
            str(SOURCE),
            str(FEATURES),
            str(TAGS),
            str(TITLES),
            str(receipt),
            *common,
        ]
    raise ValueError(model)


def run_trial(model: str, index: int, policy: dict[str, object]) -> dict[str, object]:
    rate = float(policy["learning_rates"][index])  # type: ignore[index]
    seed = int(policy["trial_seed"])
    epochs = int(policy["max_epochs_per_trial"])
    slug = model.lower()
    trial_name = f"{slug}_t{index + 1}"
    raw = SCRATCH / "hpo_raw" / f"{trial_name}.json"
    checkpoint = SCRATCH / "checkpoints" / f"hpo_p1_{trial_name}.pt"
    log = SCRATCH / "logs" / f"hpo_p1_{trial_name}.log"
    receipt = ROOT / "evidence" / f"hpo_p1_{trial_name}.json"
    if receipt.exists() or raw.exists() or checkpoint.exists() or log.exists():
        raise FileExistsError(
            f"Existing HPO trial artifact requires audit: {trial_name}"
        )
    previous = [
        ROOT / "evidence" / f"hpo_p1_{slug}_t{prior + 1}.json" for prior in range(index)
    ]
    if any(not path.is_file() for path in previous):
        raise FileNotFoundError("Earlier declared HPO trial receipt is missing")
    spent = sum(
        json.loads(path.read_text())["elapsed_wall_seconds"] for path in previous
    )
    remaining = int(policy["gpu_time_budget_seconds_per_model"] - spent)
    if remaining <= 0:
        raise RuntimeError(f"Declared per-model GPU budget exhausted: {model}")
    for directory in (raw.parent, checkpoint.parent, log.parent):
        directory.mkdir(parents=True, exist_ok=True)
    gpu_available()
    trial_config = {
        "protocol_id": policy["protocol_id"],
        "model": model,
        "trial_index": index + 1,
        "learning_rate": rate,
        "seed": seed,
        "epochs": epochs,
        "policy_sha256": digest(POLICY_PATH),
    }
    config_sha = hashlib.sha256(
        json.dumps(trial_config, sort_keys=True).encode()
    ).hexdigest()
    env = os.environ.copy()
    env["MOSAIC_HPO_LEARNING_RATE"] = str(rate)
    env["PYTHONUNBUFFERED"] = "1"
    started = now()
    start = time.perf_counter()
    args = command(model, raw, checkpoint, epochs, seed, trial_name)
    with log.open("w") as stream:
        process = subprocess.Popen(
            args,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
            returncode = 124
    elapsed = time.perf_counter() - start
    original = json.loads(raw.read_text()) if raw.exists() else {}
    passed = (
        returncode == 0
        and original.get("status")
        in {"PILOT_PASS", "LOCAL_COMMON_PROTOCOL_PILOT", "REFERENCE_PROTOCOL_PILOT"}
        and checkpoint.is_file()
    )
    result: dict[str, object] = {
        "stage_id": "E00",
        "protocol_id": policy["protocol_id"],
        "state": "DEVELOPMENT",
        "status": "HPO_TRIAL_PASS" if passed else "HPO_TRIAL_FAIL",
        "started_utc": started,
        "ended_utc": now(),
        "dataset": "microlens-100k",
        "model": model,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": rate,
        "hpo_trial_index": index + 1,
        "hpo_policy_sha256": digest(POLICY_PATH),
        "config_sha256": config_sha,
        "trial_config": trial_config,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "raw_receipt": str(raw),
        "raw_receipt_sha256": digest(raw) if raw.exists() else None,
        "log": str(log),
        "log_sha256": digest(log),
        "returncode": returncode,
        "elapsed_wall_seconds": elapsed,
        "train_seconds": original.get("train_seconds"),
        "score_seconds": original.get("score_seconds"),
        "peak_vram_bytes": original.get("peak_vram_bytes"),
        "device": original.get("device"),
        "environment": original.get("environment"),
        "input_sha256": original.get("input_sha256"),
        "validation_slices": original.get("validation_slices"),
        "ndcg_at_20": original.get("ndcg_at_20"),
        "recall_at_20": original.get("recall_at_20"),
        "training_log": original.get("training_log"),
        "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
        "checkpoint_sha256": digest(checkpoint) if checkpoint.is_file() else None,
        "warnings": [
            "Development HPO trial only; no model qualification, finalist, or locked result."
        ],
        "evidence_class": "measured_development_hpo"
        if passed
        else "measured_development_hpo_failure",
    }
    if not passed:
        result["failure_class"] = (
            "timeout_resource"
            if returncode == 124
            else "runtime_or_scientific_unclassified"
        )
        result["raw_status"] = original.get("status")
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(receipt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trial-index", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    policy = json.loads(POLICY_PATH.read_text())
    allowed = [
        model for family in policy["model_families"].values() for model in family
    ]
    if args.model not in allowed:
        raise ValueError(f"Model absent from predeclared HPO policy: {args.model}")
    indices = (
        (args.trial_index - 1,)
        if args.trial_index
        else range(int(policy["trials_per_model"]))
    )
    for index in indices:
        result = run_trial(args.model, index, policy)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "model",
                        "hpo_trial_index",
                        "status",
                        "learning_rate",
                        "elapsed_wall_seconds",
                    )
                }
            ),
            flush=True,
        )
        if result["status"] != "HPO_TRIAL_PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
