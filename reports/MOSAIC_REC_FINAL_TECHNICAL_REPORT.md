# MOSAIC-Rec technical report

Generated 2026-09-21T09:10:50.615729+00:00 from `evidence/warehouse.duckdb`.
Protocol: `mosaic-rec-2026-09-20-p1`; state: **DEVELOPMENT**.
Canonical receipts: 86; locked-execution receipts: 0.

## Scientific status

**DRAFT — no locked scientific result exists.** No finalist, frozen pair of comparators, warm noninferiority margin, or promotion decision is established.
The table below is single-seed development validation evidence only.

## Development MicroLens validation

Exact available-catalog first-event NDCG@20, using FIT-history users. Controlled-cold items have model-visible FIT collaboration hidden. These runs are not equal-budget HPO or qualified baselines.

| Model | All NDCG@20 | Controlled cold NDCG@20 | FIT-warm NDCG@20 | Receipt |
|---|---:|---:|---:|---|
| M0 | 0.012708 (n=41,694) | 0.000000 (n=918) | 0.026404 (n=20,047) | `microlens_m0_pilot` |
| M1 | 0.019197 (n=41,694) | 0.030476 (n=918) | 0.005828 (n=20,047) | `microlens_m1_pilot` |
| M2 | 0.018291 (n=41,694) | 0.030245 (n=918) | 0.000000 (n=20,047) | `microlens_m2_pilot` |
| M3 | 0.015975 (n=41,694) | 0.027928 (n=918) | 0.000000 (n=20,047) | `microlens_m3_pilot` |
| M4 | 0.015313 (n=41,694) | 0.035573 (n=918) | 0.001205 (n=20,047) | `microlens_m4_pilot` |
| M5 | 0.016732 (n=41,694) | 0.005140 (n=918) | 0.029227 (n=20,047) | `microlens_m5_pilot` |
| M6 | 0.017389 (n=41,694) | 0.005051 (n=918) | 0.030447 (n=20,047) | `microlens_m6_pilot` |
| M7 | 0.015415 (n=41,694) | 0.001485 (n=918) | 0.030546 (n=20,047) | `microlens_m7_pilot` |
| B9_BM3_MMREC_REFERENCE | 0.002922 (n=41,694) | 0.000000 (n=918) | 0.006078 (n=20,047) | `microlens_bm3_mmrec_reference_pilot` |

M4 led the single-seed controlled-cold ladder slice; M7 led FIT-warm. The table shows the cold/warm tradeoff. The BM3 reference architecture ran for one epoch and scored zero on the cold slice; it is underqualified and cannot be the sole cold comparator. FREEDOM attempt 1 was interrupted without a metric; its log and graph cache were preserved.

## Data and provenance

The three interaction tracks have four temporal role artifacts and passing mechanical leakage audits. Publisher MicroLens image/text arrays and the PixelRec publisher-image/local-text alignment passed source and item-ID QA. PixelRec raw covers remain an intact partial blocked by the official Drive quota. Raw assets and feature arrays stay in WSL scratch.

Historical content-effective timestamps are unavailable. First observed interaction time is a candidate-availability proxy, not a product launch time. Observed non-events are not verified negatives. MovieLens 32M supports behavioral replication but lacks a native image modality; its FIT-history final population is a small selected subset.

## Claims from the canonical ledger

| Claim ID | Status | Source receipt |
|---|---|---|
| `bm3_reference_pilot_cold_zero` | INTERNAL_PILOT | `microlens_bm3_mmrec_reference_pilot` |
| `core_promotion` | UNSUPPORTED | — |
| `development_ladder_tradeoff` | INTERNAL_PILOT | `microlens_m4_pilot` |
| `multi_gpu` | UNSUPPORTED | — |
| `pilot_microlens_gru4rec` | INTERNAL_PILOT | `microlens_gru4rec_pilot` |
| `pixelrec_local_content_alignment` | SUPPORTED | `pixelrec_content_alignment_p1` |
| `production_impact` | UNSUPPORTED | — |
| `reliability` | UNSUPPORTED | — |

## Pending scientific work

Baseline parity and equal-budget HPO, independent seeds, D08 finalist selection, qualification/freeze, exact locked scoring, paired uncertainty, robustness, ablations, replication, and system benchmarks are pending. The API is a local development demonstration and has no production-impact claim. See `reports/PROGRESS.md`, `reports/BLOCKERS.md`, and `reports/ADVERSARIAL_AUDIT.md` for current gates.
