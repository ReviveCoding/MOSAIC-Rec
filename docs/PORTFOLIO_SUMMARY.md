# MOSAIC-Rec portfolio summary

## The problem

Sequential recommenders can perform well for established items while failing when an item has little or no collaborative history. MOSAIC-Rec asked whether adaptive fusion of collaborative, text, image, and metadata signals could improve controlled cold-item ranking without sacrificing warm-item quality.

The primary study used MicroLens-100K with temporal FIT, VALIDATION, QUALIFICATION, and LOCKED_FINAL roles. A deterministic controlled-cold cohort had its collaborative FIT evidence hidden while permitted content remained available.

## The method

The project implemented an eight-variant MOSAIC ladder, sequence and content-capable baselines, exact full-catalog evaluation, and a single-GPU evidence pipeline. It maintained separate frozen comparators for sequence quality and inductive cold capability because one model could not fairly represent both requirements.

All model development and hyperparameter work preceded protected outcome access. Qualification froze three-seed score ensembles for BERT4Rec as the sequence comparator, a content Two-Tower as the cold comparator, and M1_T2 as the MOSAIC finalist. The locked analysis used paired user-cluster bootstrap inference with a separately frozen warm noninferiority margin.

## Evidence by phase

### Development

The M0-M7 ladder exposed a cold/warm tradeoff across variants. This was exploratory evidence used to understand candidate behavior, not a promotion result. `[development_ladder_tradeoff]`

### Qualification

Qualification applied rules frozen before outcome access to select the two comparators, finalist, seed ensembles, metrics, and statistical gates. Qualification evidence selected identities; it did not establish the final claim.

### Locked evaluation

M1_T2 exceeded the cold comparator by `0.008940` NDCG@20 in the point estimate, but the paired 95% interval crossed zero, so controlled-cold superiority was not established. `[locked_cold_point_improvement]`

M1_T2 was `0.023619` NDCG@20 below the sequence comparator on FIT-warm items and failed the frozen noninferiority gate. `[locked_warm_noninferiority_failed]`

The secondary overall comparison favored M1_T2 by `0.007482` NDCG@20, but the study design did not allow a secondary result to replace either failed co-primary gate. `[locked_overall_secondary]`

The correct terminal decision was **RETAIN_BASELINE**. `[terminal_decision]`

### Systems evidence

The project measured exact CUDA full-catalog scoring and a local FastAPI demonstration. The API completed 100 sequential requests without error at `4.49 ms` median latency on the RTX 4090 Laptop GPU. This is local demonstration evidence, not production capacity or readiness. `[systems_local_measurement]`

## What the result means

MOSAIC found a promising cold-start operating point but did not meet the uncertainty and warm-quality requirements needed for promotion. Retaining the baseline was evidence of a functioning decision process: the frozen gates prevented a favorable secondary metric from becoming a post hoc success criterion.

## Limitations

- Confirmatory robustness and ablation definitions were incomplete before protected access, so reliability-aware claims are unsupported.
- MovieLens and PixelRec received data and input preparation, but no frozen external transfer study was executed; external replication is unsupported.
- Public observational offline data do not establish causal or production impact.
- The systems study used one physical GPU and cannot support multi-GPU claims.
- First-observed interaction time was an availability proxy, and historical content revision times were unavailable.

## Portfolio takeaway

The project demonstrates recommendation modeling, multimodal feature integration, temporal experimental design, statistical decision-making, exact ranking evaluation, CUDA systems work, evidence engineering, and the discipline to preserve and report an unfavorable result.

## Claim key

Bracketed identifiers refer to supported entries in `evidence/claims.json` and `reports/CLAIM_LEDGER.md`. Limitations above mirror the ledger's unsupported claims and the authoritative final report.
