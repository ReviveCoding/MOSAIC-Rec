# MOSAIC-Rec p2 preregistration

Protocol: `mosaic-rec-2026-09-23-p2`

Created: 2026-09-23 before p2 model execution

Initial scientific state: `PILOT`

## Independence and provenance

Protocol p1 (`mosaic-rec-2026-09-20-p1`) is closed with permanent decision `RETAIN_BASELINE`. Its receipts, protected outputs, checkpoints, manifests, claim ledger, warehouse, and stage status are immutable. p2 does not append to or reinterpret p1 evidence.

The p1 post-closure analysis is hypothesis-generating only. Its manifest SHA-256 is `4c343e4892f25713a2d923f394dbfe93efe4655b952b6442b6c4b200e9695af0`. The p2 research question and slice families were chosen before any p2 VALIDATION, QUALIFICATION, or LOCKED_FINAL model outcome was observed.

## Research question

Can a support-aware expert router preserve a content expert's zero-support cold-start advantage while switching rapidly enough to a sequence expert to avoid the warm-quality collapse observed post hoc in p1?

The p1 patterns motivating this question are not p2-confirmed facts.

## Primary data and rights boundary

PixelRec50K is the primary multimodal track. Local research may use the official interactions and item attributes plus already qualified publisher-provided extracted features. Raw covers will not be extracted, transformed, or used while the rights hold remains. Raw data, publisher vectors, derived local arrays, checkpoints, and top-k outputs remain in WSL scratch and are not redistributed.

MovieLens 32M is a secondary behavioral transfer track without visual claims. MicroLens p1 protected outcomes are excluded from p2 selection and inference.

## Temporal protocol

The raw PixelRec source has 989,494 interactions, 50,000 users, 82,865 items, timestamps 1328292931 through 1656087360, and source SHA-256 `638b53ec100f760cb9bd540c361f6d6e3617c81b1c054ced63fffa41da909e4d`.

Global boundaries were fixed from the source timestamp distribution before model outcomes:

| Role | Timestamp rule |
|---|---|
| FIT source interval | `timestamp <= 1630118460` |
| VALIDATION | `1630118460 < timestamp <= 1636695517` |
| QUALIFICATION | `1636695517 < timestamp <= 1643524851` |
| LOCKED_FINAL | `1643524851 < timestamp <= 1656087360` |

All rows sharing a cutoff timestamp remain in the earlier role. Evaluated users require at least five model-visible FIT interactions before the query. The earliest eligible target event per user is used in each protected role. Candidate items require first-observed interaction timestamp at or before the query. This timestamp is an availability proxy and is not called a launch time.

## Zero-support populations

`CONTROLLED_ZERO_SUPPORT` is a deterministic 5% cohort of items observed during the source FIT interval. Selection hashes `protocol_id + ':controlled-zero:' + item_id`; all selected-item FIT interactions are removed from model-visible collaboration, while permitted content stays visible.

`NATURAL_ZERO_SUPPORT` contains non-withheld target items with no model-visible FIT interactions. It is analyzed only if first-observed candidate semantics place the target in the query catalog. Otherwise the branch is `HOLD`.

## Experts and router ladder

- `C_seq`: BERT4Rec-style sequence expert.
- `C_cold`: multimodal content Two-Tower expert.
- `Legacy`: p1-style M1_T2 architecture adapted to p2 data.
- `R1`: hard zero-support router.
- `R2`: monotonic support router with `b = softplus(raw_b)` and an exact content-only zero-support override.
- `R3`: context router using only pre-query support, age, FIT user history, recency, historical popularity/tail share, and content indicators.
- `R4`: R3 plus positive-support distillation against frozen `C_seq` scores.

Additional baselines are popularity, recent popularity, BPR-MF, LightGCN, SASRec, and VBPR where qualification is feasible. Neural families receive the same three-trial DEVELOPMENT budget, maximum 30 epochs, patience five, two GPU-hours per trial, and selection seed 20260930.

## Score alignment

Temperature scaling, per-query standardized logits, and rank-percentile normalization are compared using VALIDATION only. The fixed selection criterion is the mean of controlled-zero NDCG@20 and margin-adjusted warm NDCG@20 after unit scaling. Ties prefer temperature scaling, standardized logits, then rank-percentile normalization. The procedure and fitted values freeze before QUALIFICATION.

## Primary hypotheses

### H1: zero-support superiority

- Population: `CONTROLLED_ZERO_SUPPORT`.
- Metric: NDCG@20.
- Contrast: selected router minus `C_cold`.
- PASS: 10,000-draw paired user-cluster bootstrap 95% lower bound is greater than zero.
- Sensitivity: 10,000-draw paired item-cluster bootstrap.

### H2: warm noninferiority

- Population: FIT support greater than 20.
- Metric: NDCG@20.
- Contrast: selected router minus `C_seq`.
- Margin: `0.10 * C_seq VALIDATION warm NDCG@20`, calculated and committed before QUALIFICATION opens.
- PASS: paired user-cluster bootstrap 95% lower bound is greater than the negative frozen margin.

Promotion requires both H1 and H2. Overall NDCG@20 is secondary and cannot override either gate.

## Seeds and ensembles

HPO uses seed 20260930. Independent seeds are 20261001, 20261002, and 20261003. The HPO seed is excluded from seed qualification. No best seed is selected. Protected scoring uses the arithmetic mean of aligned scores from all three independent-seed checkpoints.

## Metrics and slices

Ranking is exact full catalog with common time-available candidates and consistent masking of model-visible training history. Primary and secondary metrics, top-k schema, support bins, popularity definitions, lifecycle bins, user cohorts, cross-slices, pairwise failure tables, and non-accuracy metrics are frozen in `configs/p2/metrics.json`.

General slices require `n >= 200`; controlled-zero slices require `n >= 50`; age-by-support cells require `n >= 200`. Counts remain visible when inference is suppressed.

## Protected access and freeze gates

QUALIFICATION remains inaccessible until baseline candidates, router family, seeds, ensemble, alignment, metrics, slices, robustness, ablations, and statistics are frozen and the protocol guard passes from a clean committed identity.

QUALIFICATION selects `C_seq`, `C_cold`, one router finalist, and calculates the numeric warm margin. E05 then hashes the entire freeze manifest. LOCKED_FINAL opens once only after the committed E05 guard passes. No protected role is used for tuning.

## Statistics, robustness, and ablations

The exact 10,000-draw bootstrap procedures, deterministic seeds, slice minimums, and Holm families are in `configs/p2/statistics.json`. Robustness transformations, intensities, populations, directions, and multiplicity are fixed in `configs/p2/robustness.json`. Ablation implementations, retraining decisions, seeds, populations, and multiplicity are fixed in `configs/p2/ablations.json`.

The project uses the term **adaptive routing**. Reliability language requires separate evidence and is prohibited by default.

## Top-k retention

QUALIFICATION and LOCKED_FINAL outputs retain compressed immutable top-20 and top-100 lists, scores, ranks, model/config identities, and router diagnostics. Every artifact is hashed. Coverage, exposure, novelty, diversity, and personalization remain secondary offline metrics.

## Decisions

- A p2 promotion requires both primary gates.
- Negative results remain valid.
- No production, causal, online engagement, external replication, reliability, superiority, warm retention, or multi-GPU claim is allowed without its exact supporting p2 evidence.
- If both gates do not pass, retain the appropriate baseline.

## Amendments

Before QUALIFICATION, a legitimate implementation clarification may create a numbered amendment that preserves this version and states whether scientific semantics changed. After QUALIFICATION access, model, split, metrics, comparators, seeds, alignment, margin rule, slice families, robustness, and ablation definitions cannot change. After LOCKED_FINAL access, any scientific change requires a new protocol identity.
