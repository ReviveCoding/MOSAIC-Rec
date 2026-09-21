# MOSAIC-Rec

MOSAIC-Rec is a temporal, multimodal recommendation study with an explicit controlled-cold item cohort. Its scientific design is in [`docs/MOSAIC_REC_SPEC.md`](docs/MOSAIC_REC_SPEC.md); the execution ledger is [`PLANS.md`](PLANS.md). The current protocol is `mosaic-rec-2026-09-20-p1` in **DEVELOPMENT**. There is no frozen finalist, qualified comparator, locked result, or promotion claim.

## Current evidence

- MicroLens-100K, MovieLens 32M, and PixelRec50K interaction tracks have temporal FIT/VALIDATION/QUALIFICATION/LOCKED_FINAL split artifacts and passing mechanical leakage audits. MicroLens has 772 controlled-cold FIT-hidden items; PixelRec has 3,109.
- The M0-M7 CUDA development ladder is complete on MicroLens. It shows a cold/warm tradeoff: M4 led the controlled-cold pilot slice while M7 led the FIT-warm pilot slice. Neither observation selects a finalist.
- A FIT-only 7/30/90-day recent-popularity development search is recorded on all three tracks. RecBole SASRec/BERT4Rec CUDA reference runtime smokes completed; they use different internal splits and are not protocol parity.
- The official PixelRec publisher image-feature file maps all 82,865 track items to finite vectors. Its image index matches the local text index exactly; aligned tag and title-presence arrays passed a separate audit. Its raw cover archive was resumed and passed full 7-Zip CRC and exact 82,865-image identity checks. Local derived-feature use remains on a rights hold; publisher features remain available for replication.
- The canonical warehouse currently validates 104 stage receipts and **zero** locked-execution receipts. FIT-only dense inputs for PixelRec and MovieLens external tracks are prepared in WSL scratch; neither is a replication result. The [technical report](reports/MOSAIC_REC_FINAL_TECHNICAL_REPORT.md) is a development draft generated from the warehouse. Check [`reports/PROGRESS.md`](reports/PROGRESS.md) for the latest live state and [`reports/RECOVERY_AUDIT.md`](reports/RECOVERY_AUDIT.md) for the post-reboot audit.

## Runtime and evidence

PowerShell 7 is the control plane; all scientific Python/PyTorch/CUDA work runs in Ubuntu-22.04 WSL. The isolated Python environment and high-I/O scratch paths are in [`configs/bootstrap.json`](configs/bootstrap.json). The sole RTX GPU is reserved for one heavy MOSAIC trainer at a time. Raw datasets, features, caches, and checkpoints stay in WSL scratch, outside Git.

The reproduction commands and protected-outcome rules are in [`docs/REPRODUCTION_RUNBOOK.md`](docs/REPRODUCTION_RUNBOOK.md). Compact stage receipts live in `evidence/`; `evidence/warehouse.duckdb` and `evidence/tables/*.parquet` are regenerated from them and checked by `mosaic_rec.validate_evidence`. Claims are tracked in `evidence/claims.json`. Dataset rights and provenance are in [`reports/DATASET_LICENSE_AUDIT.md`](reports/DATASET_LICENSE_AUDIT.md).

An optional FastAPI demonstration can load a declared, hash-verified development checkpoint on the RTX GPU and expose exact available-catalog recommendations plus model-state diagnostics. It is documented in the runbook and has not yet been profiled as a system result.

This repository currently supports a development study and reproducibility audit. It does not establish production impact, reliability, external validity, or a winning MOSAIC model.

The [final execution status](reports/FINAL_STATUS.md) is a preserved historical **BLOCKED_EXTERNAL** snapshot from the prior GPU-contention window, not current scientific closure. The GPU is reserved and execution has resumed under protocol p1 in DEVELOPMENT.





