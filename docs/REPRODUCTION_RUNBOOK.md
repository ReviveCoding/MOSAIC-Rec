# MOSAIC-Rec reproduction and recovery runbook

This runbook reproduces **the current development evidence**, not a locked scientific conclusion. It uses the project paths in `configs/bootstrap.json`: PowerShell in `C:\Users\bjw-0\Downloads\MOSAIC-Rec`, Ubuntu-22.04 WSL for all scientific Python, and `/home/bjw-0/mosaic-rec-work` for raw data, features and checkpoints. The protocol is `mosaic-rec-2026-09-20-p1` in DEVELOPMENT.

## Before any neural job

Check current GPU users and run the actual CUDA tensor probe:

```powershell
wsl -d Ubuntu-22.04 -- nvidia-smi
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python scripts/gpu_probe.py
```

One heavy MOSAIC trainer may run at a time. The advisory lock is `/home/bjw-0/mosaic-rec-work/locks/gpu.lock`. A stale PID string after reboot does not establish a live owner; verify the process and test advisory lock availability before use. Never silently use CPU for neural training, feature extraction, exact neural scoring, or robustness inference.

## Verify existing assets

`configs/data_registry.json` records official sources and raw-file hashes. `configs/split_p1.json` records the four temporal role cutoffs and controlled cold policy. The following audit checks source, split, checkpoint and feature hashes, plus Parquet footer row counts; it does not read protected outcome rows:

```powershell
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python scripts/recovery_integrity.py
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m unittest discover -s tests -v
```

The MicroLens publisher vectors and local MiniLM text vectors are in WSL scratch. PixelRec raw covers are an incomplete `.part` due to the official Google Drive quota; preserve that file. The official publisher image-feature JSON is a separate input and must pass complete-file and item-ID qualification before modeling. The source and rights boundaries are in `reports/DATASET_LICENSE_AUDIT.md`. Do not put raw datasets, covers or feature vectors in Git.

## Rebuild canonical evidence

After adding or changing any **development** receipt, rebuild from receipts and validate:

```powershell
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.warehouse evidence evidence/warehouse.duckdb evidence/tables
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.validate_evidence evidence evidence/warehouse.duckdb
```

The validator prints `locked_execution_receipts`. It must remain zero until comparator, finalist, margin, seed, metric, and robustness definitions are validly frozen. Read `reports/RECOVERY_AUDIT.md` before resuming from a host or WSL interruption.

## Development runs

Existing M0–M7 receipts and checkpoints passed the post-reboot hash audit; **do not rerun over their paths**. To reproduce a pilot independently, use a new receipt and checkpoint path, retain the seed, and label the new run as development evidence. `scripts/run_mosaic_ladder.sh` documents the historical single-GPU command sequence but writes to the existing paths and must not be replayed in place.

The independent B1 recent-popularity pilot can be rerun on a new receipt path with:

```powershell
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.popularity_split microlens-100k /home/bjw-0/mosaic-rec-work/data/microlens-100k/split_p1 evidence/microlens_recent_popularity_30d_replay.json --half-life-days 30
```

This accesses VALIDATION only. Do not inspect QUALIFICATION or LOCKED_FINAL outcome rows before the appropriate state transition. The current baseline status is in `reports/BASELINE_QUALIFICATION.md`; no public model comparison is authorized from these pilots.
