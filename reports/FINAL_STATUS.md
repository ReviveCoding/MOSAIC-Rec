# MOSAIC-Rec final execution status

Generated 2026-09-20T21:49:48.547645+00:00 from canonical evidence.

**Terminal decision for this execution window: BLOCKED_EXTERNAL (sole GPU occupied by unrelated jobs).** The scientific protocol remains in DEVELOPMENT and is **not closed**. No finalist, qualified comparator, margin, locked result, or promotion decision exists.

## Identity and evidence

- Protocol: `mosaic-rec-2026-09-20-p1`; scientific state: `DEVELOPMENT`.
- Git identity at report generation: `471d758ebed5d3499f410826ed07a5ada93aeaeb`.
- Runtime: Ubuntu-22.04; PyTorch `2.14.0+cu130`; CUDA runtime `13.0`; GPU inventory `0, NVIDIA GeForce RTX 4090 Laptop GPU, 16376 MiB, 616.92`.
- Isolated environment: `/home/bjw-0/mosaic-rec-work/.venv`; scratch: `/home/bjw-0/mosaic-rec-work`.
- `requirements.lock` SHA-256: `a46c8a71c3648ee0d3cacda594e4e99b42672fe5294f9dd334ee47523eec44f4`.
- Canonical warehouse: 55 consistent stage receipts, 8 claim rows, 0 locked-execution receipts.

## Dataset status

| Track | Current evidence | Limitation |
|---|---|---|
| MicroLens-100K | Interaction roles, FIT-hidden cold cohort, leakage audit, publisher vectors and local text features qualified | Historical content-effective time unavailable; raw-cover perturbation rights held |
| MovieLens 32M | Interaction roles and FIT-only dense behavioral input qualified | No native image modality; FIT-history final population is a selected subset |
| PixelRec50K | Interaction roles, FIT-hidden cold cohort, publisher image/local text alignment and FIT-only dense input qualified | Raw cover archive partial is quota-blocked; publisher-feature replication has not run |

## Stage dispositions

PASS below means the named pilot or preparation passed its stated contract; it does not imply scientific qualification. HOLD means a prerequisite or result is still missing.

| Stage | Disposition | Recorded evidence status |
|---|---|---|
| A00 | PASS | `PASS` |
| A01 | HOLD | `PASS_WITH_MEDIA_HOLD` |
| A02 | BLOCKED | `PARTIAL_PIXEL_COVER_BLOCKED_EXTERNAL` |
| A03 | PASS | `PASS` |
| A04 | PASS | `PASS` |
| A05 | PASS | `PASS` |
| A06 | PASS | `PASS` |
| B00 | PASS | `PASS_TEXT_ONLY` |
| B01 | PASS | `PASS_PUBLISHER_FEATURES` |
| C00 | PASS | `PILOT_PASS` |
| C01 | PASS | `PILOT_PASS` |
| C02 | PASS | `PILOT_PASS` |
| C03 | HOLD | `PILOT_PARTIAL` |
| C04 | HOLD | `PILOT_PARTIAL` |
| C05 | HOLD | `IMPORT_SMOKE_PASS` |
| D00 | PASS | `PILOT_PASS` |
| D01 | PASS | `PILOT_PASS` |
| D02 | PASS | `PILOT_PASS` |
| D03 | PASS | `PILOT_PASS` |
| D04 | PASS | `PILOT_PASS` |
| D05 | PASS | `PILOT_PASS` |
| D06 | PASS | `PILOT_PASS` |
| D07 | PASS | `PILOT_PASS` |
| D08 | HOLD | `PENDING` |
| E00 | HOLD | `PENDING` |
| E01 | HOLD | `PENDING` |
| E02 | HOLD | `PENDING` |
| E03 | HOLD | `PENDING` |
| E04 | HOLD | `PENDING` |
| E05 | HOLD | `PENDING` |
| F00 | HOLD | `PENDING` |
| F01 | HOLD | `PENDING` |
| F02 | HOLD | `PENDING` |
| F03 | HOLD | `PENDING` |
| F04 | HOLD | `PENDING` |
| F05 | HOLD | `PENDING` |
| Gxx | HOLD | `PENDING` |
| Hxx | HOLD | `PENDING` |
| I00 | HOLD | `FIT_INPUT_PREP_PASS_REPLICATION_PENDING` |
| I01 | HOLD | `FIT_INPUT_PREP_PASS_REPLICATION_PENDING` |
| I02 | HOLD | `PENDING` |
| J00 | HOLD | `PENDING` |
| J01 | HOLD | `PENDING` |
| J02 | HOLD | `PENDING` |
| J03 | HOLD | `PENDING` |
| J04 | HOLD | `PENDING` |
| J05 | HOLD | `PENDING` |
| K00 | HOLD | `PENDING` |
| K01 | HOLD | `PENDING` |
| K02 | HOLD | `DRAFT_CLAIM_LEDGER` |
| K03 | HOLD | `DRAFT_DEVELOPMENT_ONLY` |
| K04 | HOLD | `PENDING` |
| K05 | HOLD | `PENDING` |

## Scientific results available now

**Strongest qualified baseline:** none. **MOSAIC finalist:** none.
The following exact-catalog VALIDATION observations are single-seed development pilots, not confirmatory comparisons:

| Model | Slice | NDCG@20 | Users | Receipt |
|---|---|---:|---:|---|
| M4 | controlled_cold | 0.035573 | 918 | `microlens_m4_pilot` |
| M4 | warm_fit | 0.001205 | 20,047 | `microlens_m4_pilot` |
| M7 | controlled_cold | 0.001485 | 918 | `microlens_m7_pilot` |
| M7 | warm_fit | 0.030546 | 20,047 | `microlens_m7_pilot` |

The M4 cold and M7 warm pattern is an unfavorable tradeoff. No warm noninferiority or cold superiority gate can be assessed without qualified frozen comparators and paired inference. Historical SASRec/GRU4Rec cold interpretation is unqualified because their negative pool could include FIT-hidden items; their receipts remain preserved.

**Robustness:** no neural intervention result. **Ablation:** prespecified draft only. **Replication:** FIT-only inputs prepared, no model result. **Systems:** no uncontended benchmark or serving-latency result. All are HOLD.

## Claim ledger

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

Supported now: local PixelRec feature alignment and narrowly labeled internal pilot observations. Unsupported: core promotion, calibrated reliability, production impact, multi-GPU scaling, external replication and causal benefit.

## Blockers and audit

The sole RTX GPU has been used by serial unrelated ACT-Agent training, rollout and benchmark jobs. A passive five-minute idle guard repeatedly reset during phase handoffs and was stopped before this blocked report; no MOSAIC heavy job launched during the contention. PixelRec raw covers remain an intact 11.47 GB partial under the official Drive quota. MicroLens raw-cover transformation rights remain unclear. See `reports/RECOVERY_AUDIT.md`, `reports/BLOCKERS.md` and `reports/NEEDS_USER_ACTION.md`.

The final adversarial audit is **not closed**. Mechanical split/leakage checks and source mapping passed; baseline fairness, locked outcome gates, robustness, replication and system claims remain open in `reports/ADVERSARIAL_AUDIT.md`. No protected outcome was opened after recovery. This report is a blocked status, not a completed scientific result.

## Exact reproduction and resumption commands

From the Windows repository, run scientific commands in Ubuntu-22.04:

```powershell
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.validate_evidence evidence evidence/warehouse.duckdb
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m pytest -q
# After sustained free-GPU verification and inspection of existing attempt-2 outputs:
wsl -d Ubuntu-22.04 -- bash scripts/run_freedom_reference_attempt2.sh
# After that receipt is inspected and the GPU is free again:
wsl -d Ubuntu-22.04 -- bash scripts/run_mosaic_validation_replay.sh
```

Run `scripts/run_corrected_sequence_pilots.sh` and `scripts/run_bert4rec_common_pilot.sh` only as later distinct GPU jobs. Preserve every failed attempt and rebuild the warehouse after each new receipt. Do not open QUALIFICATION or LOCKED_FINAL until the protocol state and freeze gates allow it.

**Recommended next research step:** reserve a sustained exclusive GPU window; finish FREEDOM/BPR/sequence baseline pilots and checkpoint replay, then equal-budget HPO and independent seeds. If qualification succeeds, freeze comparators, margin, model and gates before any protected scoring. Retain protocol p1 unless a new scientific corruption is found.
