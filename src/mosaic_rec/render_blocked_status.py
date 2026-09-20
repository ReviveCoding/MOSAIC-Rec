"""Render an evidence-linked external-block status without claiming closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import torch


def stage_disposition(evidence_status: str) -> str:
    if "BLOCKED_EXTERNAL" in evidence_status:
        return "BLOCKED"
    if evidence_status == "PASS" or evidence_status.startswith("PILOT_PASS"):
        return "PASS"
    if evidence_status.startswith("PASS_") and "HOLD" not in evidence_status:
        return "PASS"
    return "HOLD"


def render(root: Path) -> str:
    stage = json.loads((root / "evidence/stage_status.json").read_text())
    if stage.get("execution_status") != "BLOCKED_EXTERNAL":
        raise ValueError("Blocked-status renderer requires BLOCKED_EXTERNAL execution")
    bootstrap = json.loads((root / "configs/bootstrap.json").read_text())
    con = duckdb.connect(str(root / "evidence/warehouse.duckdb"), read_only=True)
    receipts = con.execute("SELECT count(*) FROM runs").fetchone()[0]
    locked = con.execute(
        "SELECT count(*) FROM runs WHERE state='LOCKED_EXECUTION'"
    ).fetchone()[0]
    claims = con.execute(
        "SELECT claim_id,status,receipt_id FROM claims ORDER BY claim_id"
    ).fetchall()
    pilot = con.execute(
        "SELECT model,slice,value,n,receipt_id FROM metrics "
        "WHERE receipt_id IN ('microlens_m4_pilot','microlens_m7_pilot') "
        "AND metric='ndcg_at_20' AND slice IN ('controlled_cold','warm_fit') "
        "ORDER BY model,slice"
    ).fetchall()
    con.close()
    if receipts != stage["canonical_receipts_current"] or locked != 0:
        raise ValueError(
            "Blocked-status precondition: warehouse/status mismatch or locked access"
        )
    git = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    lock_hash = hashlib.sha256((root / "requirements.lock").read_bytes()).hexdigest()
    lines = [
        "# MOSAIC-Rec final execution status",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from canonical evidence.",
        "",
        (
            "**Terminal decision for this execution window: BLOCKED_EXTERNAL (sole GPU occupied by unrelated jobs).** "
            "The scientific protocol remains in DEVELOPMENT and is **not closed**. "
            "No finalist, qualified comparator, margin, locked result, or promotion decision exists."
        ),
        "",
        "## Identity and evidence",
        "",
        f"- Protocol: `{stage['protocol_id']}`; scientific state: `{stage['scientific_state']}`.",
        f"- Git identity at report generation: `{git}`.",
        f"- Runtime: {bootstrap['distro']}; PyTorch `{torch.__version__}`; CUDA runtime `{torch.version.cuda}`; GPU inventory `{bootstrap['gpu_inventory'][0]}`.",
        f"- Isolated environment: `{bootstrap['scratch_wsl']}/.venv`; scratch: `{bootstrap['scratch_wsl']}`.",
        f"- `requirements.lock` SHA-256: `{lock_hash}`.",
        f"- Canonical warehouse: {receipts} consistent stage receipts, {len(claims)} claim rows, {locked} locked-execution receipts.",
        "",
        "## Dataset status",
        "",
        "| Track | Current evidence | Limitation |",
        "|---|---|---|",
        "| MicroLens-100K | Interaction roles, FIT-hidden cold cohort, leakage audit, publisher vectors and local text features qualified | Historical content-effective time unavailable; raw-cover perturbation rights held |",
        "| MovieLens 32M | Interaction roles and FIT-only dense behavioral input qualified | No native image modality; FIT-history final population is a selected subset |",
        "| PixelRec50K | Interaction roles, FIT-hidden cold cohort, publisher image/local text alignment and FIT-only dense input qualified | Raw cover archive partial is quota-blocked; publisher-feature replication has not run |",
        "",
        "## Stage dispositions",
        "",
        "PASS below means the named pilot or preparation passed its stated contract; it does not imply scientific qualification. HOLD means a prerequisite or result is still missing.",
        "",
        "| Stage | Disposition | Recorded evidence status |",
        "|---|---|---|",
    ]
    for stage_id, evidence_status in stage["stages"].items():
        lines.append(
            f"| {stage_id} | {stage_disposition(evidence_status)} | `{evidence_status}` |"
        )
    lines.extend(
        [
            "",
            "## Scientific results available now",
            "",
            "**Strongest qualified baseline:** none. **MOSAIC finalist:** none.",
            "The following exact-catalog VALIDATION observations are single-seed development pilots, not confirmatory comparisons:",
            "",
            "| Model | Slice | NDCG@20 | Users | Receipt |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model, slice_name, value, n, receipt_id in pilot:
        lines.append(
            f"| {model} | {slice_name} | {value:.6f} | {n:,} | `{receipt_id}` |"
        )
    lines.extend(
        [
            "",
            "The M4 cold and M7 warm pattern is an unfavorable tradeoff. No warm noninferiority or cold superiority gate can be assessed without qualified frozen comparators and paired inference. Historical SASRec/GRU4Rec cold interpretation is unqualified because their negative pool could include FIT-hidden items; their receipts remain preserved.",
            "",
            "**Robustness:** no neural intervention result. **Ablation:** prespecified draft only. **Replication:** FIT-only inputs prepared, no model result. **Systems:** no uncontended benchmark or serving-latency result. All are HOLD.",
            "",
            "## Claim ledger",
            "",
            "| Claim ID | Status | Source receipt |",
            "|---|---|---|",
        ]
    )
    for claim_id, claim_status, receipt_id in claims:
        lines.append(
            f"| `{claim_id}` | {claim_status} | `{receipt_id}` |"
            if receipt_id
            else f"| `{claim_id}` | {claim_status} | — |"
        )
    lines.extend(
        [
            "",
            "Supported now: local PixelRec feature alignment and narrowly labeled internal pilot observations. Unsupported: core promotion, calibrated reliability, production impact, multi-GPU scaling, external replication and causal benefit.",
            "",
            "## Blockers and audit",
            "",
            "The sole RTX GPU has been used by serial unrelated ACT-Agent training, rollout and benchmark jobs. A passive five-minute idle guard repeatedly reset during phase handoffs and was stopped before this blocked report; no MOSAIC heavy job launched during the contention. PixelRec raw covers remain an intact 11.47 GB partial under the official Drive quota. MicroLens raw-cover transformation rights remain unclear. See `reports/RECOVERY_AUDIT.md`, `reports/BLOCKERS.md` and `reports/NEEDS_USER_ACTION.md`.",
            "",
            "The final adversarial audit is **not closed**. Mechanical split/leakage checks and source mapping passed; baseline fairness, locked outcome gates, robustness, replication and system claims remain open in `reports/ADVERSARIAL_AUDIT.md`. No protected outcome was opened after recovery. This report is a blocked status, not a completed scientific result.",
            "",
            "## Exact reproduction and resumption commands",
            "",
            "From the Windows repository, run scientific commands in Ubuntu-22.04:",
            "",
            "```powershell",
            "wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.validate_evidence evidence evidence/warehouse.duckdb",
            "wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m pytest -q",
            "# After sustained free-GPU verification and inspection of existing attempt-2 outputs:",
            "wsl -d Ubuntu-22.04 -- bash scripts/run_freedom_reference_attempt2.sh",
            "# After that receipt is inspected and the GPU is free again:",
            "wsl -d Ubuntu-22.04 -- bash scripts/run_mosaic_validation_replay.sh",
            "```",
            "",
            "Run `scripts/run_corrected_sequence_pilots.sh` and `scripts/run_bert4rec_common_pilot.sh` only as later distinct GPU jobs. Preserve every failed attempt and rebuild the warehouse after each new receipt. Do not open QUALIFICATION or LOCKED_FINAL until the protocol state and freeze gates allow it.",
            "",
            "**Recommended next research step:** reserve a sustained exclusive GPU window; finish FREEDOM/BPR/sequence baseline pilots and checkpoint replay, then equal-budget HPO and independent seeds. If qualification succeeds, freeze comparators, margin, model and gates before any protected scoring. Retain protocol p1 unless a new scientific corruption is found.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(render(args.root))
    print(args.output)


if __name__ == "__main__":
    main()
