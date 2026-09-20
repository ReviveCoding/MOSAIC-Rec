# MOSAIC-Rec execution ledger

Objective: execute the specified three-track study without opening protected outcomes before a valid freeze. Protocol identity: `mosaic-rec-2026-09-20-p1` (pilot, unsealed).

## Stage DAG and current stage

`A00-A06 -> B00-B01 -> C00-C05 -> D00-D08 -> E00-E05 -> F00-F05 -> Gxx/Hxx -> I00-I02 -> J00-J05 -> K00-K05`.

Current: C01-C05 baseline qualification and D08 finalist investigation, with PixelRec publisher-image vectors qualified for a local-only multimodal branch. Raw-cover extraction remains BLOCKED_EXTERNAL by the official Drive quota. Completed and reverified: three-track split manifests, mechanical leakage receipts, MicroLens and PixelRec publisher features, local CUDA text features, validation-only B0/B1/BPR/SASRec/GRU4Rec/LightGCN/VBPR/Two-Tower pilots, RecBole SASRec/BERT4Rec CUDA runtime smokes, MOSAIC M0-M7 development ladder. Post-reboot source/split/checkpoint/feature integrity audits passed. The canonical warehouse validates 50 receipts and zero locked-execution receipts. Pending: remaining baseline families, equal-budget HPO and seed qualification, frozen comparator/margin, locked outcomes, robustness, replication, systems, statistics, closure. A blocked branch does not block independent branches.

## Hard gates

- Source terms, citation, provenance, checksum, and schema before modeling each dataset.
- Real CUDA tensor qualification before neural workloads. One exclusive heavy GPU job via `scratch/locks/gpu.lock`.
- Temporal four-role split, leakage audit, and controlled cold cohort before model fitting.
- Baseline parity and fair HPO before comparator and finalist freeze.
- Config, checkpoint, seed, margin, metrics, and robustness definitions frozen before locked outcomes.
- Claims derive from canonical evidence only.

## GPU queue

Completed CUDA jobs: an earlier BPR-MF pilot, SASRec, GRU4Rec, LightGCN, VBPR, content Two-Tower pilots, revision-pinned text extraction, RecBole SASRec/BERT4Rec reference smokes, MOSAIC M0-M7 pilots, and a one-epoch BM3 reference runtime pilot. FREEDOM attempt 1 interrupted after graph-cache creation without a metric; preserve it. Queue: distinct FREEDOM attempt 2, common FIT-visible BPR-MF pilot, M1/M4/M6/M7 development validation replay and seed stability, qualified HPO, exact ranking, robustness, systems benchmarks. GPU lock path: `/home/bjw-0/mosaic-rec-work/locks/gpu.lock`; the old pre-reboot PID text was superseded by reference pilot owners and the advisory lock is now free. An unrelated ACT-Agent SFT trainer currently holds material VRAM/compute, so heavy MOSAIC jobs wait. CPU acquisition/preprocessing/statistics can overlap safely.

## Artifacts and validation

Windows repository holds code, compact evidence, reports, config manifests and aggregate outputs. `/home/bjw-0/mosaic-rec-work` holds raw datasets, feature caches, checkpoints, and logs. Validate by checksums, schema and leakage checks, known-answer metric tests, CUDA tensor probe, stage receipts, evidence and claim-ledger audits.

## Recovery

See `reports/RECOVERY_AUDIT.md` for the verified post-reboot state. Resume the existing 11.47 GB PixelRec `cover.7z` partial with source-verified gdown continuation; retain failed logs, never overwrite a locked protocol, and create a new protocol identity after any post-lock scientific change. Do not claim completion until all stages have terminal status and `reports/FINAL_STATUS.md` exists.
