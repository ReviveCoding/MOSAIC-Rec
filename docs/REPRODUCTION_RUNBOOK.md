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

PixelRec's local image/text/metadata alignment can be reproduced only to new output paths; the existing derived arrays and B01 receipt must be preserved. The current `evidence/pixelrec_content_alignment_p1.json` records matching item order, dimensions, finiteness, 109 tag categories, 192 missing titles, and all source/output hashes. `scripts/recovery_integrity.py` verifies those arrays without loading protected outcome rows.

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

`scripts/run_mosaic_validation_replay.sh` is a read-only CUDA replay of the existing M1/M4/M6/M7 development checkpoints. It writes first-event per-user exact available-catalog outcomes to WSL scratch, checks aggregate agreement with the original receipts within 1e-6, and creates distinct D08 replay receipts. It never retrains and has no QUALIFICATION or LOCKED_FINAL input. Run it only after the GPU is free and the earlier reference-pilot queue is resolved.

The declared MMRec BM3 and FREEDOM one-epoch reference runtime pilots use the isolated checkout in WSL scratch and the hash-qualified MicroLens publisher vectors. After checking that unrelated GPU compute processes have exited, run them sequentially:

```powershell
wsl -d Ubuntu-22.04 -- bash scripts/run_mmrec_reference_pilots.sh
```

The script refuses existing receipt/checkpoint paths and checks GPU occupancy before each job. It preserves separate logs and failure receipts. An epoch-one pilot is never a qualified comparator. Rebuild and validate the evidence warehouse after inspecting any new receipt.

`src/mosaic_rec/statistics.py` implements paired user-cluster percentile intervals, cold item-cluster sensitivity, and Holm adjustment. `configs/statistics_development_p1.json` records the proposed design. It is **not** an E04 scientific freeze; no locked scores may be analyzed until the full qualification manifest is sealed.

## Local demonstration API

The optional FastAPI service is a **development checkpoint demonstration**. It uses the hash-verified M1 pilot checkpoint at the FIT-end snapshot and scores the available catalog on CUDA. It requires an otherwise free GPU and holds the exclusive project GPU lock for its lifetime; do not run it beside a trainer. In Ubuntu-22.04 WSL, from the repository root:

```bash
MOSAIC_SERVING_MANIFEST=configs/serving_development_p1.json \
  /home/bjw-0/mosaic-rec-work/.venv/bin/uvicorn mosaic_rec.serving:app \
  --host 127.0.0.1 --port 8000
```

`GET /health` reports the loaded model identity. `POST /recommend` accepts `{"user_id": 1, "k": 20}`. `POST /diagnostics` returns actual branch gate weights and score contributions from the same model state. An unconfigured service returns HTTP 503 for ranking, and a user without FIT history returns 404. The API does not simulate a production system or report scientific performance; no J04 latency result is claimed before a separate benchmark.
