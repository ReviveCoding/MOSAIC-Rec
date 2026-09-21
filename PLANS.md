# MOSAIC-Rec execution ledger

2026-09-21 current boundary: CUDA validation replay and three-seed M1/M4/M6/M7 development stability completed. E00 equal-budget HPO and baseline qualification are in progress under `configs/hpo_development_p1.json`, one model/trial at a time; inspect receipts and rebuild the warehouse after each model. PixelRec raw `cover.7z` completed and passed SHA-256, full CRC, and 82,865-item mapping, while local raw-cover feature generation remains rights-held. E01-E05 and all locked stages remain pending; locked execution receipts remain zero.

2026-09-21 resumption: sole RTX GPU reserved for MOSAIC. Continue protocol `mosaic-rec-2026-09-20-p1` in DEVELOPMENT. Current queue order is FREEDOM distinct attempt 2; rebuild and validate warehouse; M1/M4/M6/M7 CUDA validation replay with 1e-6 agreement gate; common FIT-visible BPR; corrected SASRec/GRU4Rec/BERT4Rec; additional-seed stability; qualified baseline HPO and seed qualification; frozen comparator/finalist/margin/analysis manifest; locked execution only after protocol guard. Inspect each job before the next. The prior BLOCKED_EXTERNAL commit/report remains historical evidence.

Objective: execute the specified three-track study without opening protected outcomes before a valid freeze. Protocol identity: `mosaic-rec-2026-09-20-p1` (pilot, unsealed).

## Stage DAG and current stage

`A00-A06 -> B00-B01 -> C00-C05 -> D00-D08 -> E00-E05 -> F00-F05 -> Gxx/Hxx -> I00-I02 -> J00-J05 -> K00-K05`.

Current: E00 equal-budget HPO and C05 reference parity remain active. Completed: three-track temporal splits and leakage audits; MicroLens and PixelRec publisher features; local CUDA text features; baseline pilots; RecBole SASRec/BERT4Rec runtime smokes; MOSAIC M0-M7 development ladder; M1/M4/M6/M7 CUDA validation replay and three-seed development stability; corrected BPR, SASRec/GRU4Rec, BERT4Rec, and LightGCN pilots. PixelRec raw-cover acquisition and integrity passed, with derived-feature rights held. FIT-only PixelRec and MovieLens adapters are prepared for later replication. No finalist, frozen comparator, noninferiority margin, protected outcome, or promotion exists. Pending: remaining equal-budget HPO, parity/seed qualification, scientific freeze, locked outcomes, robustness, ablations, replication, systems, statistics, closure.

## Hard gates

- Source terms, citation, provenance, checksum, and schema before modeling each dataset.
- Real CUDA tensor qualification before neural workloads. One exclusive heavy GPU job via `scratch/locks/gpu.lock`.
- Temporal four-role split, leakage audit, and controlled cold cohort before model fitting.
- Baseline parity and fair HPO before comparator and finalist freeze.
- Config, checkpoint, seed, margin, metrics, and robustness definitions frozen before locked outcomes.
- Claims derive from canonical evidence only.

## GPU queue

Completed CUDA jobs: an earlier BPR-MF pilot, SASRec, GRU4Rec, LightGCN, VBPR, content Two-Tower pilots, revision-pinned text extraction, RecBole SASRec/BERT4Rec reference smokes, MOSAIC M0-M7 pilots, and a one-epoch BM3 reference runtime pilot. FREEDOM attempt 1 interrupted after graph-cache creation without a metric; preserve it. Completed after the reserved GPU window: FREEDOM attempt 2, corrected FIT-visible BPR-MF and sequence pilots, M1/M4/M6/M7 CUDA validation replay, and additional-seed development stability. Equal-budget 30-epoch CUDA HPO has terminal grids for B2/B3/B4/B5/B6/B8; B7 trial 1 is active, then B9/B10/M1/M4/M6/M7 remain. Next: HPO summary, reference parity and seed qualification, then QUALIFICATION-only comparator/finalist/margin selection, freeze, guarded locked execution, robustness, systems and reporting. GPU lock path: `/home/bjw-0/mosaic-rec-work/locks/gpu.lock`; the old pre-reboot PID text was superseded by reference pilot owners and the advisory lock is now free. Unrelated ACT-Agent serial GPU phases blocked the queue. Passive watcher session 27209 was stopped before the blocked final-status snapshot; resume only with renewed free-GPU checks and output-path inspection. CPU acquisition/preprocessing/statistics can overlap safely.

## Artifacts and validation

Windows repository holds code, compact evidence, reports, config manifests and aggregate outputs. `/home/bjw-0/mosaic-rec-work` holds raw datasets, feature caches, checkpoints, and logs. Validate by checksums, schema and leakage checks, known-answer metric tests, CUDA tensor probe, stage receipts, evidence and claim-ledger audits. Render the technical report with `python -m mosaic_rec.render_technical_report evidence/warehouse.duckdb evidence/stage_status.json reports/MOSAIC_REC_FINAL_TECHNICAL_REPORT.md` from the isolated WSL environment after warehouse updates.

## Recovery

See `reports/RECOVERY_AUDIT.md` for the verified post-reboot state. The 11.47 GB PixelRec partial was resumed and the complete archive qualified on 2026-09-21; retain failed logs and the earlier BLOCKED_EXTERNAL commit/report. Never overwrite a locked protocol; create a new identity after any post-lock scientific change. Do not claim completion until all stages have terminal status and the current `reports/FINAL_STATUS.md` reflects canonical closure.

