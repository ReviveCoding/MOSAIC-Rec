# MOSAIC-Rec execution ledger

2026-09-23 terminal boundary: protocol `mosaic-rec-2026-09-20-p1` is CLOSED with `RETAIN_BASELINE`. All feasible stages have terminal PASS, FAIL, HOLD, or INVALID dispositions. The primary locked co-gates failed; Gxx/Hxx are invalid due incomplete pre-lock definitions; replication is LOCAL_VALIDITY_ONLY; systems is a partial pass with ANN held. Canonical evidence validates 169 receipts and the final audit is complete. Any further scientific work requires a new preregistered protocol that preserves p1.

2026-09-23 current boundary: E00 HPO and the frozen E01 independent-seed queue are complete in DEVELOPMENT. E01 contains 21 effective seed runs across seven candidates, plus the preserved BERT4Rec CUDA infrastructure failure and its linked audited attempt 2. The canonical warehouse contains 137 receipts and no locked execution. Current work is the required read-only B5/B6/B7 common-slice replay, followed by the guarded transition to QUALIFICATION for E02-E04. No comparator, finalist, margin, or protected outcome has been selected.

2026-09-21 current boundary: CUDA validation replay and three-seed M1/M4/M6/M7 development stability completed. E00 equal-budget HPO and baseline qualification are in progress under `configs/hpo_development_p1.json`, one model/trial at a time; inspect receipts and rebuild the warehouse after each model. PixelRec raw `cover.7z` completed and passed SHA-256, full CRC, and 82,865-item mapping, while local raw-cover feature generation remains rights-held. E01-E05 and all locked stages remain pending; locked execution receipts remain zero.

2026-09-21 resumption: sole RTX GPU reserved for MOSAIC. Continue protocol `mosaic-rec-2026-09-20-p1` in DEVELOPMENT. Current queue order is FREEDOM distinct attempt 2; rebuild and validate warehouse; M1/M4/M6/M7 CUDA validation replay with 1e-6 agreement gate; common FIT-visible BPR; corrected SASRec/GRU4Rec/BERT4Rec; additional-seed stability; qualified baseline HPO and seed qualification; frozen comparator/finalist/margin/analysis manifest; locked execution only after protocol guard. Inspect each job before the next. The prior BLOCKED_EXTERNAL commit/report remains historical evidence.

Objective: execute the specified three-track study without opening protected outcomes before a valid freeze. Protocol identity: `mosaic-rec-2026-09-20-p1` (pilot, unsealed).

## Stage DAG and current stage

`A00-A06 -> B00-B01 -> C00-C05 -> D00-D08 -> E00-E05 -> F00-F05 -> Gxx/Hxx -> I00-I02 -> J00-J05 -> K00-K05`.

Current: E01 is closed as `E01_SEED_QUALIFICATION_COMPLETE_UNQUALIFIED`; the required sequence common-slice replay is active before E02. Completed: three-track temporal splits and leakage audits; MicroLens and PixelRec publisher features; local CUDA text features; baseline pilots; RecBole SASRec/BERT4Rec runtime smokes; MOSAIC M0-M7 development ladder; M1/M4/M6/M7 CUDA validation replay and development stability; equal-budget E00 HPO; and the frozen E01 three-seed queue. PixelRec raw-cover acquisition and integrity passed, with derived-feature rights held. FIT-only PixelRec and MovieLens adapters are prepared for later replication. No finalist, frozen comparator, noninferiority margin, protected outcome, or promotion exists. Pending: common-slice replay, QUALIFICATION-only E02-E04 decisions, E05 freeze, guarded locked outcomes, robustness, ablations, replication, systems, statistics, and closure.

## Hard gates

- Source terms, citation, provenance, checksum, and schema before modeling each dataset.
- Real CUDA tensor qualification before neural workloads. One exclusive heavy GPU job via `scratch/locks/gpu.lock`.
- Temporal four-role split, leakage audit, and controlled cold cohort before model fitting.
- Baseline parity and fair HPO before comparator and finalist freeze.
- Config, checkpoint, seed, margin, metrics, and robustness definitions frozen before locked outcomes.
- Claims derive from canonical evidence only.

## GPU queue

Completed CUDA jobs include the full equal-budget E00 grid and all 21 effective E01 independent-seed runs. FREEDOM attempt 1 remains interrupted and preserved; BERT4Rec E01 seed 20260925 attempt 1 remains a preserved CUDA infrastructure failure linked to its successful distinct attempt 2. Next GPU work is the read-only B5/B6/B7 validation slice replay, one checkpoint at a time, then QUALIFICATION-only evaluation after an explicit state transition. GPU lock path: `/home/bjw-0/mosaic-rec-work/locks/gpu.lock`. ContextWeave launched during two E01 transitions; both already-running MOSAIC seeds finished, its child exited, and its supervisor is reversibly paused for the user-reserved MOSAIC GPU window. CPU preprocessing, evidence validation, and reporting may overlap safely.

## Artifacts and validation

Windows repository holds code, compact evidence, reports, config manifests and aggregate outputs. `/home/bjw-0/mosaic-rec-work` holds raw datasets, feature caches, checkpoints, and logs. Validate by checksums, schema and leakage checks, known-answer metric tests, CUDA tensor probe, stage receipts, evidence and claim-ledger audits. Render the technical report with `python -m mosaic_rec.render_technical_report evidence/warehouse.duckdb evidence/stage_status.json reports/MOSAIC_REC_FINAL_TECHNICAL_REPORT.md` from the isolated WSL environment after warehouse updates.

## Recovery

See `reports/RECOVERY_AUDIT.md` for the verified post-reboot state. The 11.47 GB PixelRec partial was resumed and the complete archive qualified on 2026-09-21; retain failed logs and the earlier BLOCKED_EXTERNAL commit/report. Never overwrite a locked protocol; create a new identity after any post-lock scientific change. Do not claim completion until all stages have terminal status and the current `reports/FINAL_STATUS.md` reflects canonical closure.



