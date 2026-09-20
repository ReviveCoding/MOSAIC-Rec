# MOSAIC-Rec Codex Operating Contract

Read `docs/MOSAIC_REC_SPEC.md` and
`MASTER_PROMPT_MOSAIC_REC.md` completely before major work.

## Runtime

Windows PowerShell is the control plane.

All substantive scientific Python/CUDA/PyTorch execution must run in
Ubuntu-22.04 under WSL.

Read:
- configs/bootstrap.json
- configs/gpu_policy.json

for canonical paths and GPU policy.

## GPU

This project must not silently become CPU-only.

When CUDA qualification succeeds, run these workloads on CUDA where supported:

- multimodal embedding extraction
- BPR neural training when applicable
- LightGCN
- Two-Tower
- GRU4Rec
- SASRec
- BERT4Rec
- VBPR
- BM3
- FREEDOM or equivalent neural multimodal baseline
- every MOSAIC neural model
- neural full-catalog ranking
- neural robustness inference

One physical GPU means one large training process at a time.

Use GPU parallelism through:
- batched CUDA kernels
- BF16/AMP when qualified
- pinned memory
- asynchronous/non-blocking host-to-device copies
- parallel DataLoader workers
- persistent workers/prefetch when measured beneficial
- cached content embeddings
- batched exact full-catalog scoring

Run CPU preprocessing, DuckDB/Polars work, bootstrap statistics,
and reporting concurrently when scientifically safe.

Do not claim multi-GPU scaling without physical multi-GPU measurement.

## Scientific state

PILOT
→ DEVELOPMENT
→ QUALIFICATION
→ FROZEN
→ LOCKED_EXECUTION
→ ANALYSIS
→ CLOSED

No post-lock:
- model reselection
- split changes
- threshold changes
- seed changes
- metric changes
- comparator changes

Scientific changes after protected access require a new protocol identity.

## Data

Before use:
- official source
- current terms/license
- provenance
- hash
- redistribution rules
- citation
- schema qualification

Do not use TMDB content for model training/features.

Never fabricate unavailable datasets.

## Claims

Separate:
- measured
- derived
- simulated
- projected
- external
- unsupported/unexecuted

Negative findings are valid findings.

Do not claim public/offline benchmark performance as production impact.

## Execution discipline

Maintain:

reports/PROGRESS.md
reports/BLOCKERS.md
reports/NEEDS_USER_ACTION.md
reports/DECISIONS.md
evidence/stage_status.json

Do not stop after scaffolding or smoke tests.

Continue every feasible independent stage until
reports/FINAL_STATUS.md exists.