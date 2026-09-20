"""Render a source-linked technical report from canonical warehouse rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def render(warehouse: Path, stage_status: Path) -> str:
    status = json.loads(stage_status.read_text())
    con = duckdb.connect(str(warehouse), read_only=True)
    runs = con.execute("SELECT count(*) FROM runs").fetchone()[0]
    locked = con.execute(
        "SELECT count(*) FROM runs WHERE state='LOCKED_EXECUTION'"
    ).fetchone()[0]
    claims = con.execute(
        "SELECT claim_id, status, receipt_id FROM claims ORDER BY claim_id"
    ).fetchall()
    metric_rows = con.execute(
        "SELECT receipt_id, model, slice, value, n FROM metrics "
        "WHERE dataset='microlens-100k' AND role='VALIDATION' "
        "AND metric='ndcg_at_20' AND receipt_id IN "
        "('microlens_m0_pilot','microlens_m1_pilot','microlens_m2_pilot',"
        "'microlens_m3_pilot','microlens_m4_pilot','microlens_m5_pilot',"
        "'microlens_m6_pilot','microlens_m7_pilot',"
        "'microlens_bm3_mmrec_reference_pilot') "
        "AND slice IN ('all','controlled_cold','warm_fit') "
        "ORDER BY model, slice"
    ).fetchall()
    con.close()
    metrics: dict[str, dict[str, tuple[float, int]]] = {}
    receipts: dict[str, str] = {}
    for receipt, model, slice_name, value, n in metric_rows:
        metrics.setdefault(model, {})[slice_name] = (value, n)
        receipts[model] = receipt
    ladder = {
        model: metrics[model]
        for model in (f"M{i}" for i in range(8))
        if model in metrics
    }
    cold_leader = (
        max(ladder, key=lambda model: ladder[model]["controlled_cold"][0])
        if ladder
        else None
    )
    warm_leader = (
        max(ladder, key=lambda model: ladder[model]["warm_fit"][0]) if ladder else None
    )
    lines = [
        "# MOSAIC-Rec technical report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from `evidence/warehouse.duckdb`.",
        f"Protocol: `{status['protocol_id']}`; state: **{status['scientific_state']}**.",
        f"Canonical receipts: {runs}; locked-execution receipts: {locked}.",
        "",
        "## Scientific status",
        "",
    ]
    if locked == 0:
        lines.extend(
            [
                (
                    "**DRAFT — no locked scientific result exists.** No finalist, frozen pair of "
                    "comparators, warm noninferiority margin, or promotion decision is established."
                ),
                "The table below is single-seed development validation evidence only.",
                "",
            ]
        )
    lines.extend(
        [
            "## Development MicroLens validation",
            "",
            (
                "Exact available-catalog first-event NDCG@20, using FIT-history users. "
                "Controlled-cold items have model-visible FIT collaboration hidden. "
                "These runs are not equal-budget HPO or qualified baselines."
            ),
            "",
            "| Model | All NDCG@20 | Controlled cold NDCG@20 | FIT-warm NDCG@20 | Receipt |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for model in (
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "B9_BM3_MMREC_REFERENCE",
    ):
        row = metrics.get(model, {})

        def value(name: str, values: dict[str, tuple[float, int]] = row) -> str:
            return (
                f"{values[name][0]:.6f} (n={values[name][1]:,})"
                if name in values
                else "—"
            )

        rid = receipts.get(model, "")
        lines.append(
            f"| {model} | {value('all')} | {value('controlled_cold')} | "
            f"{value('warm_fit')} | `{rid}` |"
        )
    lines.extend(
        [
            "",
            (
                f"{cold_leader} led the single-seed controlled-cold ladder slice; "
                f"{warm_leader} led FIT-warm. The table shows the cold/warm tradeoff. The BM3 reference architecture "
                "ran for one epoch and scored zero on the cold slice; it is underqualified and "
                "cannot be the sole cold comparator. FREEDOM attempt 1 was interrupted without "
                "a metric; its log and graph cache were preserved."
            ),
            "",
            "## Data and provenance",
            "",
            (
                "The three interaction tracks have four temporal role artifacts and passing mechanical "
                "leakage audits. Publisher MicroLens image/text arrays and the PixelRec publisher-image/local-text "
                "alignment passed source and item-ID QA. PixelRec raw covers remain an intact partial "
                "blocked by the official Drive quota. Raw assets and feature arrays stay in WSL scratch."
            ),
            "",
            (
                "Historical content-effective timestamps are unavailable. First observed interaction "
                "time is a candidate-availability proxy, not a product launch time. Observed non-events "
                "are not verified negatives. MovieLens 32M supports behavioral replication but lacks a "
                "native image modality; its FIT-history final population is a small selected subset."
            ),
            "",
            "## Claims from the canonical ledger",
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
            "## Pending scientific work",
            "",
            (
                "Baseline parity and equal-budget HPO, independent seeds, D08 finalist selection, "
                "qualification/freeze, exact locked scoring, paired uncertainty, robustness, ablations, "
                "replication, and system benchmarks are pending. The API is a local development "
                "demonstration and has no production-impact claim. See `reports/PROGRESS.md`, "
                "`reports/BLOCKERS.md`, and `reports/ADVERSARIAL_AUDIT.md` for current gates."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("warehouse", type=Path)
    parser.add_argument("stage_status", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(render(args.warehouse, args.stage_status))
    print(args.output)


if __name__ == "__main__":
    main()
