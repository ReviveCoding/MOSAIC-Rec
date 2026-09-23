# MOSAIC-Rec interview narrative

## Two-minute version

I built MOSAIC-Rec to study a common recommendation tradeoff: content can help items with no collaborative history, but a model still has to preserve quality for established items.

I designed a temporal protocol with separate FIT, VALIDATION, QUALIFICATION, and LOCKED_FINAL roles. The controlled-cold cohort had no model-visible FIT collaboration. I developed an eight-variant multimodal sequence ladder, qualified a sequence comparator and a content-capable cold comparator, and froze a three-seed MOSAIC finalist, exact full-catalog metrics, a warm noninferiority margin, and paired bootstrap inference before opening final outcomes.

The locked result was mixed. M1_T2 improved the cold point estimate by `0.008940` NDCG@20, but the confidence interval crossed zero, so superiority was not established. `[locked_cold_point_improvement]` It was `0.023619` below BERT4Rec on warm items and failed noninferiority. `[locked_warm_noninferiority_failed]` The overall secondary metric improved by `0.007482`, but I did not substitute that result for the failed primary gates. `[locked_overall_secondary]`

I closed the protocol with **RETAIN_BASELINE**. `[terminal_decision]` That was the correct decision because the study required both credible cold improvement and acceptable warm quality. The project demonstrates how I combine recommendation modeling, statistical inference, GPU systems work, and disciplined claims.

## Five-minute technical version

### Problem formulation

The central question was whether multimodal item content could supply an inductive fallback for cold items without degrading a strong sequence model on warm items. I treated those as two separate requirements and used two comparators: BERT4Rec for sequence quality and a content Two-Tower for cold capability.

### Experimental design

The interaction data was split chronologically into four roles. FIT trained the models, VALIDATION supported development and HPO, QUALIFICATION selected frozen identities, and LOCKED_FINAL supplied the final protected evidence. A deterministic item cohort had FIT collaboration hidden while permitted content remained visible.

The model ladder covered fixed fusion, learned fusion, support-conditioned gating, behavioral dropout, modality dropout, consistency signaling, and a monotonic-gate challenger. Development exposed a real cold/warm tradeoff rather than a single dominant architecture. `[development_ladder_tradeoff]`

Qualification froze B7_BERT4REC_T3 as the sequence comparator, B4_TWO_TOWER_T3 as the cold comparator, and M1_T2 as the finalist. Each locked model was a three-seed score ensemble. Evaluation used exact full-catalog ranking, paired user-cluster bootstrap inference, and a precomputed warm noninferiority margin.

### Result and decision

On controlled cold items, M1_T2's point estimate exceeded the Two-Tower by `0.008940`, but its paired interval included zero. `[locked_cold_point_improvement]` On warm items, M1_T2 was `0.023619` below BERT4Rec and failed noninferiority. `[locked_warm_noninferiority_failed]` M1_T2 also improved the overall secondary metric by `0.007482`. `[locked_overall_secondary]`

The co-primary rule required cold superiority and warm noninferiority. Both gates failed, so `RETAIN_BASELINE` followed directly from the frozen protocol. `[terminal_decision]`

### Systems and audit

I ran the neural work on a single RTX GPU, serialized heavy jobs with a lock, and used batching and cached content features. I maintained hashed checkpoints, stage receipts, and a canonical DuckDB/Parquet warehouse. A local FastAPI demonstration completed 100 sequential requests without error at `4.49 ms` median latency, which I report only as local systems evidence. `[systems_local_measurement]`

The final audit also found that robustness and ablation families had been named but not defined precisely enough before protected access. I marked them invalid instead of filling in details after seeing the result. External inputs were prepared, but no frozen transfer study was executed, so I do not claim replication.

## Deep-dive version

### Why the evaluation needed two comparators

An ID-based sequence model can be strong for warm items yet structurally unable to represent a FIT-hidden cold item. A content-capable model can score that item but may be weaker on mature collaborative patterns. Comparing MOSAIC with only one of them would make one half of the claim artificially easy. Separate comparators made the promotion criterion match the research question.

### Why exact full-catalog scoring mattered

Sampled negatives can make ranking quality sensitive to the sampling distribution. Qualification and final evaluation therefore scored the available catalog exactly. The candidate-time rule used first-observed interaction time as a disclosed availability proxy, and previously seen user items were excluded.

### Why the finalist was selected before final access

Development produced several cold/warm operating points. Qualification used a frozen balancing rule to select M1_T2 rather than choosing the best final slice after seeing protected outcomes. Three-seed score ensembling also prevented best-seed selection.

### Why `RETAIN_BASELINE` was scientifically correct

The cold point estimate alone was insufficient because the uncertainty interval crossed zero. `[locked_cold_point_improvement]` The warm deficit was much larger than the frozen noninferiority allowance. `[locked_warm_noninferiority_failed]` The favorable overall result answered a secondary question and could not replace either primary requirement. `[locked_overall_secondary]` Retaining the baseline followed the decision rule and avoided outcome-driven reinterpretation. `[terminal_decision]`

### What failed outside the primary result

The confirmatory robustness and ablation plans lacked exact intensities, scopes, implementations, and multiplicity mappings. After protected access, adding those details would have been a protocol change informed by the result. The project therefore labels those stages invalid and avoids reliability-aware claims.

MovieLens and PixelRec received provenance, split, leakage, feature, and FIT-input work. Their neural transfer identities and external statistics were not frozen, so external replication is unsupported. The appropriate next step is a new preregistered protocol, not a post hoc extension of p1.

### What I would change next

I would formulate training around the measured cold/warm frontier, give the gate an observable modality-quality signal, freeze every robustness intensity and ablation implementation, and preregister PixelRec transfer before any external outcome access. I would preserve p1 as the baseline evidence rather than rewriting its history.

## Likely interviewer questions

### Did MOSAIC work?

It showed a positive cold point estimate and a positive overall secondary result, but it did not pass the frozen promotion gates. Cold superiority was not established, and warm noninferiority failed. The decision was `RETAIN_BASELINE`. `[locked_cold_point_improvement]` `[locked_warm_noninferiority_failed]` `[terminal_decision]`

### Why not promote based on the overall metric?

Overall NDCG@20 was secondary and mixes support regimes. The research goal required improvement for controlled cold items without unacceptable warm loss. Replacing that rule after seeing outcomes would be post hoc. `[locked_overall_secondary]`

### Why was BERT4Rec not the cold comparator?

Its learned item representation depends on FIT interaction support, so it cannot fairly represent the controlled inductive requirement by itself. The protocol used a separate content-capable Two-Tower comparator.

### How did you prevent leakage?

I used temporal roles, FIT-only training histories and negative eligibility, a deterministic FIT-hidden cold cohort, candidate-time masking, cross-role audits, and guards that blocked protected access until the freeze was complete.

### What was the most important negative finding?

The finalist's cold-start benefit did not generalize into acceptable warm behavior. The warm deficit was `0.023619` NDCG@20 against BERT4Rec and failed the frozen margin. `[locked_warm_noninferiority_failed]`

### Can you call the model reliability-aware?

No. The confirmatory intervention protocol was underspecified, and gate coefficients are not calibrated reliability probabilities. Reliability remains unsupported.

### Did the result replicate?

No external neural replication was executed under a frozen transfer protocol. The result is local to p1.

### Was this production tested?

No. The FastAPI result is a local development demonstration. It completed 100 sequential requests without error at `4.49 ms` median latency, but it does not establish production capacity, readiness, or impact. `[systems_local_measurement]`

### What does the project demonstrate if the model was not promoted?

It demonstrates problem formulation, baseline fairness, multimodal and sequence modeling, protected experimentation, paired inference, exact ranking, GPU execution, evidence engineering, and the ability to make and defend an unfavorable decision.

## Claim discipline

Bracketed identifiers map to supported claims in the final ledger. Do not describe p1 as proving cold superiority, warm retention, reliability, external replication, production impact, causal benefit, or multi-GPU scaling.
