# p2 progress

Protocol: `mosaic-rec-2026-09-23-p2`

Scientific state: `DEVELOPMENT`

## Current phase

Crash recovery complete. P2-D03 R3 attempt 3 completed under the exclusive GPU lock; P2-D04 R4 is the next scientific job.

## Completed

- Verified p1 Git and artifact hashes without modifying p1.
- Rechecked current official PixelRec and MovieLens terms and source locations.
- Rechecked current RecBole/MMRec repository identities and primary-environment CUDA operation.
- Preregistered roles, cohorts, model families, primary gates, metrics, top-k retention, statistics, robustness, ablations, seed policy, and protected-access gates.
- Committed the preregistration boundary at Git `6f00e42` before p2 model execution.
- Materialized and hashed fresh PixelRec roles under `/home/bjw-0/mosaic-rec-work/p2`.
- Created a distinct deterministic controlled-zero cohort with 3,013 items.
- Passed temporal-boundary, controlled-zero, natural-zero, count, null, and duplicate audits.
- Preserved two failed preparation attempts rather than overwriting them.
- Created p2-owned feature paths for 82,865 items and verified aligned indices, expected image/text dimensions, and finite values.
- Excluded all seven engagement aggregates from the permitted metadata inputs.
- Built FIT-only item and user context tables and hashed earliest eligible role query manifests.
- Confirmed zero candidate-time violations across 27,918 validation, 26,900 qualification, and 25,367 locked query rows.
- Preserved the first BERT4Rec and Two-Tower attempts but invalidated their validation metrics after detecting incomplete FIT-history masking for 905 users.
- Corrected the shared loader to retain complete FIT histories for candidate masking while keeping last-50 neural inputs.
- Completed exact CPU global and 30-day recent popularity baselines; the optimized scorer preserved exact ranks and top-100 lists.
- R1 attempt 1 was blocked before model loading by an unrelated ACT-Agent CUDA trainer; no p2 router artifact exists yet.
- Added CUDA pilot implementations for BPR-MF, LightGCN, SASRec, and VBPR; corrected the shared zero-based item indexing contract before executing them.
- Rebuilt the p2-only canonical warehouse with invalidation-aware selection semantics: 19 p2 receipts, 8 valid metric rows, zero p1 canonical receipts, and zero protected access. The two invalid expert receipts remain stored but their 8 metric rows are excluded.
- Completed and independently rehashed corrected attempt-2 expert runs after a sustained idle GPU window. C_seq validation NDCG@20 is 0.001864 overall and 0.013602 warm; C_cold is 0.001312 overall, 0.000940 controlled zero, 0.001411 natural zero, and 0.001570 warm.
- R1 hard routing preserved warm quality at 0.013539 but scored zero on controlled zero under per-query standardized alignment. This unfavorable development result is retained and motivates the preregistered alignment comparison.
- R2 monotonic routing reached 0.014892 warm NDCG@20 and 0.002306 overall but also scored zero on controlled zero. R3 attempt 1 then stopped before model loading when ACT-Agent PID 137901 began a new CUDA SFT job.
- Recovered after the reboot from pre-reboot HEAD `49ee02aa3da271fc1a0dec3350ea8f526d142785`. Preserved and validated the untracked preregistered slice implementation and tests; Ruff, targeted tests, and the full 71-test suite pass.
- Reverified all corrected C_seq, C_cold, R1, and R2 hashes and Parquet schemas. The canonical warehouse passes with 25 p2 receipts, 24 metric rows, zero p1 receipts, and zero protected access.
- The reboot audit found no R3, R4, HPO, temporary, or queue artifact. The R3 attempt-1 receipt remains a prestart contention event with no scientific output. The stale lock text names completed R2 PID 137697, while an actual `flock` probe was available.
- After reboot the RTX 4090 Laptop GPU showed 0 MiB, 0% utilization, no CUDA process, and passed the real torch CUDA/BF16 tensor qualification.
- R3 attempt 2 completed the queue idle window, but ACT-Agent launched CUDA PID 2404 during the runner's independent idle guard. The p2 guard failed closed before lock acquisition/model loading; only the 760-byte prestart log exists and is hashed. Any retry will use attempt 3 paths.
- R3 attempt 3 waited through ACT-Agent's follow-on evaluation, acquired the p2 lock only after six idle checks, and completed on the RTX 4090 Laptop GPU. Exact VALIDATION NDCG@20 was 0.001939 overall, 0 controlled zero, 0.000131 natural zero, and 0.013539 warm. The unfavorable zero-support result is preserved; checkpoint and output hashes independently match the receipt.
- Rebuilt the p2-only warehouse after R3: 27 receipts, 28 valid metric rows, zero p1 receipts, zero QUALIFICATION access, zero LOCKED_FINAL access, validation PASS.

## Active next action

Execute P2-D04 R4 attempt 1 with the unchanged preregistered ladder after a fresh GPU/process/lock check; inspect and canonicalize it before baseline pilots.

## Protected access counters

- QUALIFICATION model outcomes: 0
- LOCKED_FINAL model outcomes: 0

## Current blockers

- Raw-cover extraction and modification remain on rights hold.
- Official publisher text-feature download is quota-blocked. Amendment 001 selects the already authorized local MiniLM embedding derived from official title/tag/description fields; its p2 copy and alignment qualification passed.
- No active GPU contention remains from the completed ACT-Agent evaluation at the R3 boundary; every subsequent heavy job must repeat the live process and lock checks.
