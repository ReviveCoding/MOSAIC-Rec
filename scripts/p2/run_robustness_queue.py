"""Crash-safe serial executor for the frozen p2 robustness cells."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mosaic_rec.p2.contracts import load_json, sha256_file


def _stem(intervention: str, intensity: float) -> str:
    return f"{intervention.lower()}_{intensity:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    cells = load_json(args.repo / "configs/p2/e04_analysis_manifest.json")[
        "robustness"
    ]["cells"]
    log_dir = args.scratch / "robustness/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for position, cell in enumerate(cells, start=1):
        intervention = str(cell["intervention"])
        intensity = float(cell["intensity"])
        stem = _stem(intervention, intensity)
        receipt_path = args.repo / f"evidence/p2/p2-g00-{stem}.json"
        output_path = args.scratch / f"robustness/{stem}_topk.parquet"
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            if (
                receipt.get("status") != "PASS"
                or receipt.get("intervention") != intervention
                or float(receipt.get("intensity")) != intensity
                or not output_path.is_file()
                or sha256_file(output_path) != receipt["locked_final"]["output_sha256"]
            ):
                raise RuntimeError(f"Invalid terminal robustness cell: {stem}")
            print(f"[{position}/{len(cells)}] validated and skipped {stem}", flush=True)
            continue
        if output_path.exists():
            raise RuntimeError(f"Receipt-less partial robustness output: {output_path}")
        log_path = log_dir / f"{stem}.log"
        if log_path.exists():
            raise RuntimeError(
                f"Receipt-less robustness log requires audit: {log_path}"
            )
        command = [
            sys.executable,
            str(args.repo / "scripts/p2/run_robustness_cell.py"),
            intervention,
            str(intensity),
            "--repo",
            str(args.repo),
            "--scratch",
            str(args.scratch),
            "--lock",
            str(args.lock),
        ]
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=args.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Robustness cell failed; audit {log_path}")
        receipt = load_json(receipt_path)
        if sha256_file(output_path) != receipt["locked_final"]["output_sha256"]:
            raise RuntimeError(f"Post-run robustness hash mismatch: {stem}")
        print(f"[{position}/{len(cells)}] PASS {stem}", flush=True)


if __name__ == "__main__":
    main()
