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

`mosaic_rec.protocol` defines the allowed PILOT → DEVELOPMENT → QUALIFICATION → FROZEN → LOCKED_EXECUTION → ANALYSIS → CLOSED transitions and fails closed on protected role access. A freeze manifest must identify source, split/cohort/feature hashes, both comparator roles, finalist, seeds, HPO selection, metric/candidate definitions, margin, gates, robustness, and ablations. This repository is still in DEVELOPMENT; there is no valid freeze manifest and `LOCKED_FINAL` access is prohibited.

## Development runs

Existing M0–M7 receipts and checkpoints passed the post-reboot hash audit; **do not rerun over their paths**. To reproduce a pilot independently, use a new receipt and checkpoint path, retain the seed, and label the new run as development evidence. `scripts/run_mosaic_ladder.sh` documents the historical single-GPU command sequence but writes to the existing paths and must not be replayed in place.

The independent B1 recent-popularity pilot can be rerun on a new receipt path with:

```powershell
wsl -d Ubuntu-22.04 -- /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.popularity_split microlens-100k /home/bjw-0/mosaic-rec-work/data/microlens-100k/split_p1 evidence/microlens_recent_popularity_30d_replay.json --half-life-days 30
```

This accesses VALIDATION only. Do not inspect QUALIFICATION or LOCKED_FINAL outcome rows before the appropriate state transition. The current baseline status is in `reports/BASELINE_QUALIFICATION.md`; no public model comparison is authorized from these pilots.

`scripts/run_mosaic_validation_replay.sh` is a read-only CUDA replay of the existing M1/M4/M6/M7 development checkpoints. It writes first-event per-user exact available-catalog outcomes to WSL scratch, checks aggregate agreement with the original receipts within 1e-6, and creates distinct D08 replay receipts. It never retrains and has no QUALIFICATION or LOCKED_FINAL input. Run it only after the GPU is free and the earlier reference-pilot queue is resolved.

The MMRec BM3 one-epoch reference pilot is complete. FREEDOM attempt 1 is preserved as interrupted; do not replay it over its paths. Attempt 2 uses the same hash-qualified publisher vectors through a distinct WSL input/cache path. After confirming sustained absence of unrelated GPU work, launch only the prepared retry:

```powershell
wsl -d Ubuntu-22.04 -- bash scripts/run_freedom_reference_attempt2.sh
```

The script refuses existing attempt-2 receipt/checkpoint/cache paths and checks GPU occupancy. If it exits abruptly without a receipt, preserve its log/cache and create a distinct interrupted receipt before any further attempt. An epoch-one pilot is never a qualified comparator. Rebuild and validate the evidence warehouse after inspecting any new receipt.

The historical SASRec/GRU4Rec pilots may have trained on FIT-hidden cold item IDs as sampled negatives. Their receipts are preserved and their cold-start interpretation is unqualified. `scripts/run_corrected_sequence_pilots.sh` declares distinct, event-time-safe development reruns under `configs/sequence_coldguard_pilot_p1.json`. It requires a free GPU and writes new checkpoint, log and receipt paths; execute it only after the earlier GPU queue and inspect each result before warehouse refresh. These reruns remain unqualified pilots without fair HPO, reference parity and independent seeds.

`scripts/run_bert4rec_common_pilot.sh` declares a local CUDA Cloze pilot with a distinct receipt and checkpoint under `configs/bert4rec_common_pilot_p1.json`. It checks GPU occupancy, holds the exclusive project lock, uses FIT_VISIBLE for training, and scores the first VALIDATION event per FIT-history user with exact available-catalog ranking. FIT-hidden cold IDs are excluded from training targets and softmax classes. The pilot has not run; even a successful run would remain unqualified until RecBole parity, fair HPO and seed replication.

`src/mosaic_rec/statistics.py` implements paired user-cluster percentile intervals, cold item-cluster sensitivity, and Holm adjustment. `configs/statistics_development_p1.json` records the proposed design. It is **not** an E04 scientific freeze; no locked scores may be analyzed until the full qualification manifest is sealed.

`src/mosaic_rec/interventions.py` defines a seeded item derangement for cached image-embedding mismatch. Its permutation has no fixed points and does not alter the vector multiset. It is a feature-level mismatch intervention, not a claim about raw-image blur, crop, or occlusion; those remain held by the license audit.

## PixelRec FIT-only replication input

`evidence/pixelrec_fit_dense_adapter_p1.json` records the prepared FIT-only dense index, which aligns item row `i` with the qualified publisher image and local text feature row `i-1` (padding is ID zero). Its data files are in `/home/bjw-0/mosaic-rec-work/data/pixelrec50k/fit_dense_p1/`, outside Git. Every dense user/item ID and timestamp was independently reversed to the source FIT_VISIBLE Parquet. The adapter accessed no validation or protected outcome rows and does not establish a replication result. Do not rerun over its output path; use a distinct path and receipt identity for a reproduction.

`evidence/movielens_fit_dense_adapter_p1.json` records the FIT-only MovieLens behavioral input in `/home/bjw-0/mosaic-rec-work/data/ml-32m/fit_dense_p1/`. DuckDB bounded memory to 4 GB while mapping 22,400,143 FIT rows to dense user IDs and all official `movies.csv` item IDs. An independent full-multiset reverse mapping against FIT_VISIBLE found zero differences. The mapped Parquet and user/item maps are hash-identified; no validation or protected outcome rows entered preparation. This is an input artifact, not evidence of behavioral replication.

## Local demonstration API

The optional FastAPI service is a **development checkpoint demonstration**. It uses the hash-verified M1 pilot checkpoint at the FIT-end snapshot and scores the available catalog on CUDA. It requires an otherwise free GPU and holds the exclusive project GPU lock for its lifetime; do not run it beside a trainer. In Ubuntu-22.04 WSL, from the repository root:

```bash
MOSAIC_SERVING_MANIFEST=configs/serving_development_p1.json \
  /home/bjw-0/mosaic-rec-work/.venv/bin/uvicorn mosaic_rec.serving:app \
  --host 127.0.0.1 --port 8000
```

`GET /health` reports the loaded model identity. `POST /recommend` accepts `{"user_id": 1, "k": 20}`. `POST /diagnostics` returns actual branch gate weights and score contributions from the same model state. An unconfigured service returns HTTP 503 for ranking, and a user without FIT history returns 404. The API does not simulate a production system or report scientific performance; no J04 latency result is claimed before a separate benchmark.
