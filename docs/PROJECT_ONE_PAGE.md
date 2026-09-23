# MOSAIC-Rec

## Multimodal sequential recommendation under cold-start constraints

### Problem

Recommendation models often depend on collaborative history, which creates a difficult tradeoff between new-item discovery and quality for established items. MOSAIC-Rec tested whether adaptive fusion of collaborative, text, image, and metadata signals could improve controlled cold-item ranking while retaining warm-item quality.

### Approach

- Built temporal FIT, VALIDATION, QUALIFICATION, and LOCKED_FINAL roles.
- Hid FIT collaboration for a deterministic controlled-cold item cohort.
- Implemented an eight-variant multimodal sequence ladder and qualified separate sequence and cold-capable comparators.
- Froze model identities, three-seed ensembles, exact full-catalog metrics, a warm noninferiority margin, and bootstrap inference before protected evaluation.
- Ran neural workloads and exact scoring on an RTX 4090 Laptop GPU with canonical DuckDB/Parquet evidence and preserved failures.

### Locked result

| Decision component | Result |
|---|---|
| Cold point difference vs. content Two-Tower | `+0.008940` NDCG@20; 95% interval crossed zero `[locked_cold_point_improvement]` |
| Warm difference vs. BERT4Rec | `-0.023619` NDCG@20; noninferiority failed `[locked_warm_noninferiority_failed]` |
| Overall secondary difference vs. BERT4Rec | `+0.007482` NDCG@20 `[locked_overall_secondary]` |
| Final decision | **RETAIN_BASELINE** `[terminal_decision]` |

The model showed a useful cold-start signal but did not satisfy the uncertainty and warm-quality requirements for promotion. The positive overall result remained secondary.

### Systems and engineering

The project included CUDA feature extraction, training, exact ranking, crash recovery, immutable receipts, checkpoint hashing, an exclusive GPU lock, and a local FastAPI demonstration. The API completed 100 sequential requests without error at `4.49 ms` median latency; this is local demonstration evidence rather than production capacity. `[systems_local_measurement]`

### Research rigor

- Preserved unsuccessful models, infrastructure failures, and unfavorable results.
- Kept protected outcomes out of HPO and model selection.
- Marked robustness and confirmatory ablations invalid when their definitions proved incomplete.
- Held external replication claims because no frozen transfer study was executed.
- Separated offline scientific evidence from causal and production claims.

### Skills demonstrated

Recommendation systems, personalization, cold start, multimodal learning, temporal evaluation, leakage control, hyperparameter studies, statistical inference, PyTorch/CUDA, full-catalog ranking, ML systems, FastAPI, DuckDB/Parquet evidence engineering, and reproducible research.

### Scope

The conclusion is local to the frozen MicroLens protocol. It does not establish reliability-aware behavior, external replication, production impact, causal benefit, or multi-GPU scaling.
