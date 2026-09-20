# Crash recovery audit — 2026-09-20

Protocol `mosaic-rec-2026-09-20-p1` remains in DEVELOPMENT. Ubuntu-22.04 restarted after an unexpected Windows reboot; no MOSAIC process survived. Git HEAD remains `946f2334b64da3751d271ed6f68d4513ba97926d`; prior project changes remain uncommitted and were not discarded.

## Verified state

- Existing WSL environment passed a new real CUDA tensor check: PyTorch 2.14.0+cu130, one RTX 4090 Laptop GPU, 16 GB VRAM, BF16 available, finite matrix multiplication. Receipt: `evidence/gpu_requalification_20260920.json`.
- Read-only WSL integrity audit matched **49** recorded source, split, checkpoint, and feature hashes. Parquet footer row counts matched; no QUALIFICATION or LOCKED_FINAL outcome rows were read. Receipt: `evidence/recovery_integrity_20260920.json`.
- M0–M7 receipts and checkpoint hashes are intact. The cold/warm tradeoff remains unchanged. Both RecBole SASRec and BERT4Rec reference runtime smokes have completed receipts. No finalist, qualified comparator, margin, or promotion gate is frozen.
- The pre-reboot warehouse was one completed BERT4Rec receipt and one claim entry behind. The old database was preserved as `evidence/warehouse_pre_recovery.duckdb` (SHA-256 `789743062bcbd19bd746dd674ff9e1d52a5df98460a7ac631d1e0785e34ee79`). Rebuilt warehouse validation passed: **38** stage receipts, **6** claim rows, **0** locked-execution receipts.
- The GPU lock file still contains pre-reboot PID 20631. That PID does not exist, and a nonblocking advisory lock acquisition succeeded. The file was retained; the operating-system lock is free. No checkpoint or partial output was removed.

## Interrupted work and safe resume point

- PixelRec `cover.7z7a52zyqj.part` is **11,470,372,864 bytes**, has a valid 7z header, and is unchanged since the pre-reboot transfer stopped. The official PixelRec50K folder lists `cover.7z` as Google Drive file `1VlQxEUoFntgQx0WnNcRsgJu_EtYEBzlS`. The installed gdown 6.4.0 documents `--continue` and retries. Resume this exact file into the existing destination; verify final hash/archive structure before feature use.
- Two subsequent continuation attempts against that official file returned Google's shared-file download-quota error, with and without saved cookies. The partial remained unchanged. Visual PixelRec work is BLOCKED_EXTERNAL pending restored official access.
- The last completed MOSAIC model stage is D07 (M7). The earliest interrupted branch is A02 PixelRec cover acquisition. C05 baseline reference runtime smoke completed, but common-protocol parity and baseline qualification remain incomplete. D08 finalist selection has not begun. No stage previously recorded as RUNNING was promoted to PASS.
- A separate ContextWeave process appeared after the audit and used about 9 GB of the sole GPU. Do not launch a heavy MOSAIC neural job until GPU contention clears. This is a transient resource hold, not a CPU-fallback authorization.

No protected outcome metric was accessed during recovery. The protocol identity does not need replacement.
