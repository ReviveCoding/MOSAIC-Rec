# p2 progress

Protocol: `mosaic-rec-2026-09-23-p2`

Scientific state: `DEVELOPMENT`

## Current phase

P2-D01 R1 hard zero-support router pilot after completed expert pilots.

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
- Completed the BERT4Rec CUDA pilot: warm NDCG@20 0.013161; both zero-support slices 0.
- Completed the multimodal Two-Tower CUDA pilot: controlled-zero NDCG@20 0.000845 and warm 0.001637.
- Persisted both exact validation top-100 artifacts and checkpoint hashes.
- Completed exact CPU global and 30-day recent popularity baselines; the optimized scorer preserved exact ranks and top-100 lists.
- R1 attempt 1 was blocked before model loading by an unrelated ACT-Agent CUDA trainer; no p2 router artifact exists yet.
- Built the p2-only canonical warehouse: 18 p2 receipts, 16 metric rows, zero p1 canonical receipts, and zero protected access.

## Active next action

While ACT-Agent occupies the GPU, build and validate the p2-only canonical warehouse and remaining router/baseline code. Retry R1 only after a sustained idle window.

## Protected access counters

- QUALIFICATION model outcomes: 0
- LOCKED_FINAL model outcomes: 0

## Current blockers

- Raw-cover extraction and modification remain on rights hold.
- Official publisher text-feature download is quota-blocked. Amendment 001 selects the already authorized local MiniLM embedding derived from official title/tag/description fields; p2 copy/alignment qualification is pending.
- An old shared p1 advisory lock names PID 79333, which is absent; p2 will use its own lock after documenting stale-lock handling. No p2 heavy job has started.
