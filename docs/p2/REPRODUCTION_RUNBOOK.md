# P2 reproduction runbook

P2 is closed. Validate existing evidence; do not rerun protected scoring.

1. Use Ubuntu-22.04 and `/home/bjw-0/mosaic-rec-work/.venv`.
2. Verify p1 immutable hashes in `src/mosaic_rec/p2/contracts.py`.
3. Verify the E05 manifest hash and every receipt/output SHA-256.
4. Rebuild the p2 warehouse with the command in `reports/p2/FINAL_STATUS.md`.
5. Run Ruff, p2 tests, and `git diff --check`.

Large data, checkpoints, and top-k artifacts remain in `/home/bjw-0/mosaic-rec-work/p2`.
