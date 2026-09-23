# MOSAIC-Rec

MOSAIC-Rec is a temporal multimodal recommendation study with controlled FIT-hidden cold items, separate sequence and cold-capable comparators, exact full-catalog evaluation, and canonical evidence. The scientific design is in [docs/MOSAIC_REC_SPEC.md](docs/MOSAIC_REC_SPEC.md).

Protocol `mosaic-rec-2026-09-20-p1` is **CLOSED** with terminal decision **RETAIN_BASELINE**. The M1_T2 finalist improved the controlled-cold point estimate and the secondary overall metric, but cold superiority was not established and warm noninferiority failed. External replication remains HOLD, so the result is local to the MicroLens protocol.

## Key artifacts

- [Final status](reports/FINAL_STATUS.md)
- [Technical report](reports/MOSAIC_REC_FINAL_TECHNICAL_REPORT.md)
- [Final adversarial audit](reports/ADVERSARIAL_AUDIT.md)
- [Claim ledger](reports/CLAIM_LEDGER.md)
- [Reproduction runbook](docs/REPRODUCTION_RUNBOOK.md)
- [Dataset and license audit](reports/DATASET_LICENSE_AUDIT.md)
- [Execution ledger](PLANS.md)

## Evidence summary

- MicroLens-100K, MovieLens 32M, and PixelRec50K have temporal split and leakage evidence.
- The canonical DuckDB/Parquet warehouse validates 169 receipts, 772 metric rows, 4 bootstrap intervals, and 13 claims.
- Frozen MicroLens comparators were B7_BERT4REC_T3 and B4_TWO_TOWER_T3; the finalist was M1_T2.
- LOCKED_FINAL M1_T2 NDCG@20 was 0.012936 on controlled cold and 0.003447 on FIT-warm.
- The frozen co-primary gates both failed, so no model promotion or reliability claim is supported.
- Robustness and confirmatory ablations are invalid because material definitions were incomplete before protected access.
- Local systems evidence includes exact GPU scoring and a FastAPI development demonstration; it does not establish production readiness.

## Runtime

PowerShell is the control plane. Scientific Python, PyTorch, and CUDA run in Ubuntu-22.04 under WSL using `/home/bjw-0/mosaic-rec-work/.venv`; large data, features, caches, checkpoints, logs, and per-user locked outputs remain in WSL scratch. Neural jobs use the sole RTX 4090 Laptop GPU one at a time.

Validate the repository from PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec && PYTHONPATH=src /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.validate_evidence evidence evidence/warehouse.duckdb'
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec && PYTHONPATH=src /home/bjw-0/mosaic-rec-work/.venv/bin/python -m pytest -q'
```

Raw datasets and feature arrays are not redistributed. Offline public data do not support production impact, causal, or multi-GPU claims.
