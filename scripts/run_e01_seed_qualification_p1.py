"""Run frozen E01 post-HPO independent-seed qualification jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_hpo_p1 as hpo

MANIFEST = ROOT / "configs/e01_seed_qualification_p1.json"
SCRATCH = Path("/home/bjw-0/mosaic-rec-work")

RAW_DIR = SCRATCH / "e01_raw"
CHECKPOINT_DIR = SCRATCH / "checkpoints"
LOG_DIR = SCRATCH / "logs"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_model(config_id: str) -> str:
    if config_id.startswith("B7_BERT4REC"):
        return "B7_BERT4REC"
    if config_id.startswith("B4_TWO_TOWER"):
        return "B4_TWO_TOWER"
    if config_id.startswith("M1_"):
        return "M1"
    if config_id.startswith("M4_"):
        return "M4"
    if config_id.startswith("M6_"):
        return "M6"
    raise ValueError(f"Unsupported E01 candidate: {config_id}")


def candidates(manifest: dict) -> list[dict]:
    return [
        manifest["sequence_candidate"],
        manifest["cold_baseline_candidate"],
        *manifest["mosaic_candidates"],
    ]


def validate_source(candidate: dict) -> tuple[Path, dict]:
    path = ROOT / candidate["source_receipt"]
    if not path.is_file():
        raise FileNotFoundError(path)

    receipt = json.loads(path.read_text())

    model = candidate_model(candidate["config_id"])

    if receipt["model"] != model:
        raise ValueError(
            f"Source model mismatch for {candidate['config_id']}: "
            f"{receipt['model']} != {model}"
        )

    if float(receipt["learning_rate"]) != float(candidate["learning_rate"]):
        raise ValueError(f"Source LR mismatch: {path}")

    if int(receipt["epochs"]) != int(candidate["epochs"]):
        raise ValueError(f"Source epoch mismatch: {path}")

    if receipt["status"] != "HPO_TRIAL_PASS":
        raise ValueError(f"Source HPO receipt is not PASS: {path}")

    return path, receipt


def validate_existing(
    receipt_path: Path,
    manifest_hash: str,
    config_id: str,
    seed: int,
    attempt: int = 1,
) -> dict:
    result = json.loads(receipt_path.read_text())

    if (
        result.get("stage_id") != "E01"
        or result.get("config_id") != config_id
        or result.get("seed") != seed
        or result.get("e01_manifest_sha256") != manifest_hash
        or int(result.get("attempt", 1)) != attempt
    ):
        raise ValueError(f"Existing E01 receipt identity mismatch: {receipt_path}")

    if result.get("status") not in {"E01_SEED_PASS", "E01_SEED_FAIL"}:
        raise ValueError(f"Existing E01 receipt is not terminal: {receipt_path}")

    return result


def infrastructure_retry_eligible(result: dict) -> bool:
    """Permit only the audited CUDA_ERROR_UNKNOWN failure for attempt 2."""
    raw_path = Path(str(result.get("raw_receipt") or ""))
    if (
        result.get("status") != "E01_SEED_FAIL"
        or result.get("returncode") == 0
        or result.get("checkpoint") is not None
        or result.get("validation_slices") is not None
        or not raw_path.is_file()
    ):
        return False
    raw = json.loads(raw_path.read_text())
    exception = str(raw.get("exception") or "")
    return raw.get("status") == "FAIL_LOCAL_PILOT" and (
        "CUDA_ERROR_UNKNOWN" in exception or "CUDA error: unknown error" in exception
    )


def run_one(
    candidate: dict,
    seed: int,
    manifest: dict,
    manifest_hash: str,
    attempt: int = 1,
) -> dict:
    config_id = candidate["config_id"]
    model = candidate_model(config_id)

    source_path, source = validate_source(candidate)

    slug = config_id.lower()
    base_run_id = f"e01_seedqual_p1_{slug}_seed{seed}"
    if attempt not in {1, 2}:
        raise ValueError("Only the original E01 attempt and audited attempt 2 exist")
    run_id = base_run_id if attempt == 1 else f"{base_run_id}_attempt{attempt}"

    raw = RAW_DIR / f"{run_id}.json"
    checkpoint = CHECKPOINT_DIR / f"{run_id}.pt"
    log = LOG_DIR / f"{run_id}.log"
    receipt = ROOT / "evidence" / f"{run_id}.json"

    if receipt.is_file():
        result = validate_existing(
            receipt,
            manifest_hash,
            config_id,
            seed,
            attempt,
        )
        if attempt == 1 and result["status"] == "E01_SEED_FAIL":
            retry = ROOT / "evidence" / f"{base_run_id}_attempt2.json"
            if retry.is_file():
                retry_result = validate_existing(
                    retry, manifest_hash, config_id, seed, 2
                )
                if (
                    retry_result.get("status") != "E01_SEED_PASS"
                    or retry_result.get("retry_of_receipt")
                    != str(receipt.relative_to(ROOT))
                    or retry_result.get("retry_of_receipt_sha256") != digest(receipt)
                ):
                    raise ValueError(f"Invalid E01 infrastructure retry: {retry}")
                print(
                    json.dumps(
                        {
                            "run_id": retry.stem,
                            "status": retry_result["status"],
                            "action": "SKIP_EXISTING_INFRASTRUCTURE_RETRY",
                        }
                    ),
                    flush=True,
                )
                return retry_result
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": result["status"],
                    "action": "SKIP_EXISTING_TERMINAL",
                }
            ),
            flush=True,
        )
        return result

    retry_of_receipt = None
    retry_of_receipt_sha256 = None
    if attempt == 2:
        base_receipt = ROOT / "evidence" / f"{base_run_id}.json"
        if not base_receipt.is_file():
            raise FileNotFoundError(
                "Attempt 2 requires the preserved attempt-1 receipt"
            )
        base_result = validate_existing(base_receipt, manifest_hash, config_id, seed, 1)
        if not infrastructure_retry_eligible(base_result):
            raise ValueError(
                "Attempt-1 evidence is not eligible for infrastructure retry"
            )
        retry_of_receipt = str(base_receipt.relative_to(ROOT))
        retry_of_receipt_sha256 = digest(base_receipt)

    partials = [path for path in (raw, checkpoint, log) if path.exists()]
    if partials:
        raise FileExistsError(
            "Partial E01 artifacts require audit before retry: "
            + ", ".join(map(str, partials))
        )

    for directory in (RAW_DIR, CHECKPOINT_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    hpo.gpu_available()

    learning_rate = float(candidate["learning_rate"])
    epochs = int(candidate["epochs"])

    args = hpo.command(
        model,
        raw,
        checkpoint,
        epochs,
        seed,
        run_id,
    )

    env = os.environ.copy()
    env["MOSAIC_HPO_LEARNING_RATE"] = str(learning_rate)
    env["PYTHONUNBUFFERED"] = "1"

    # Infrastructure timeout only; this is not a scientific selection budget.
    timeout_seconds = 7200

    started = now()
    runner_git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    start = time.perf_counter()

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
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = 124

    elapsed = time.perf_counter() - start

    original = json.loads(raw.read_text()) if raw.exists() else {}

    accepted_raw_status = {
        "PILOT_PASS",
        "LOCAL_COMMON_PROTOCOL_PILOT",
        "REFERENCE_PROTOCOL_PILOT",
    }

    passed = (
        returncode == 0
        and original.get("status") in accepted_raw_status
        and checkpoint.is_file()
    )

    source_hash = digest(source_path)

    result = {
        "stage_id": "E01",
        "protocol_id": manifest["protocol_id"],
        "state": "DEVELOPMENT",
        "status": "E01_SEED_PASS" if passed else "E01_SEED_FAIL",
        "started_utc": started,
        "ended_utc": now(),
        "dataset": "microlens-100k",
        "config_id": config_id,
        "model": model,
        "seed": seed,
        "attempt": attempt,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "source_hpo_receipt": str(source_path.relative_to(ROOT)),
        "source_hpo_receipt_sha256": source_hash,
        "source_hpo_checkpoint_sha256": source.get("checkpoint_sha256"),
        "e01_manifest": str(MANIFEST.relative_to(ROOT)),
        "e01_manifest_sha256": manifest_hash,
        "source_hpo_closure_commit": manifest["source_hpo_closure_commit"],
        "runner_git_commit": runner_git_commit,
        "retry_of_receipt": retry_of_receipt,
        "retry_of_receipt_sha256": retry_of_receipt_sha256,
        "retry_reason": (
            "transient_cuda_error_unknown_after_successful_cuda_requalification"
            if attempt == 2
            else None
        ),
        "raw_receipt": str(raw),
        "raw_receipt_sha256": digest(raw) if raw.is_file() else None,
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
        "checkpoint_sha256": (digest(checkpoint) if checkpoint.is_file() else None),
        "warnings": [
            "Independent DEVELOPMENT seed qualification only.",
            "QUALIFICATION and LOCKED_FINAL remain unopened.",
            "No comparator or MOSAIC finalist is selected by this receipt.",
        ],
        "evidence_class": (
            "measured_development_seed_qualification"
            if passed
            else "measured_development_seed_qualification_failure"
        ),
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

    print(
        json.dumps(
            {
                "run_id": run_id,
                "config_id": config_id,
                "seed": seed,
                "status": result["status"],
                "elapsed_wall_seconds": elapsed,
            }
        ),
        flush=True,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--retry-attempt", type=int, choices=(2,))
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run/skip the complete frozen E01 queue serially.",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())

    manifest_hash = digest(MANIFEST)

    hpo_summary = ROOT / manifest["source_hpo_summary"]
    if not hpo_summary.is_file():
        raise FileNotFoundError(hpo_summary)

    if digest(hpo_summary) != manifest["source_hpo_summary_sha256"]:
        raise ValueError("Frozen E01 source HPO summary hash mismatch")

    source_commit = manifest["source_hpo_closure_commit"]

    subprocess.check_call(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=ROOT,
    )

    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestor_check.returncode != 0:
        raise ValueError(
            "Frozen E01 source HPO closure commit is not an ancestor of HEAD"
        )

    allowed_seeds = [int(seed) for seed in manifest["independent_seed_policy"]["seeds"]]

    candidate_map = {item["config_id"]: item for item in candidates(manifest)}

    if args.all:
        if (
            args.candidate is not None
            or args.seed is not None
            or args.retry_attempt is not None
        ):
            raise ValueError("--all cannot be combined with --candidate/--seed")

        for candidate in candidates(manifest):
            for seed in allowed_seeds:
                result = run_one(
                    candidate,
                    seed,
                    manifest,
                    manifest_hash,
                )
                if result["status"] != "E01_SEED_PASS":
                    raise SystemExit(
                        f"E01 queue stopped on {candidate['config_id']} seed {seed}"
                    )
        return

    if args.candidate is None or args.seed is None:
        raise ValueError("Specify --candidate CONFIG_ID --seed SEED, or use --all")

    if args.candidate not in candidate_map:
        raise ValueError(f"Candidate absent from frozen E01 manifest: {args.candidate}")

    if args.seed not in allowed_seeds:
        raise ValueError(f"Seed absent from frozen E01 manifest: {args.seed}")

    result = run_one(
        candidate_map[args.candidate],
        args.seed,
        manifest,
        manifest_hash,
        attempt=args.retry_attempt or 1,
    )

    if result["status"] != "E01_SEED_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
