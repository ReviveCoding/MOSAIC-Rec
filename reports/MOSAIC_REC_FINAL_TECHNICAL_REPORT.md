# MOSAIC-Rec technical report

## Study design

Protocol `mosaic-rec-2026-09-20-p1` tested adaptive multimodal sequential recommendation on a four-role temporal MicroLens-100K design: FIT, VALIDATION, QUALIFICATION, and LOCKED_FINAL. Controlled cold items had no model-visible FIT collaboration. QUALIFICATION selected two separate comparators and one MOSAIC finalist under rules frozen before access. LOCKED_FINAL used exact full-catalog scoring and three-seed score ensembles.

MovieLens 32M and PixelRec50K received provenance, temporal split, leakage, and FIT-input preparation. They did not receive valid neural replication because the external transfer protocol was not completely frozen before the primary protected result.

## Development and qualification

The M0-M7 ladder produced an unfavorable cold/warm tradeoff. Across independent DEVELOPMENT seeds, M4_T3 had the strongest mean controlled-cold result, while B7_T3 and warm-oriented MOSAIC variants were stronger on FIT-warm items. The frozen qualification rule selected M1_T2 by minimizing its worse ordinal rank across qualification cold and warm NDCG@20.

Frozen identities were B7_BERT4REC_T3 as `C_seq`, B4_TWO_TOWER_T3 as `C_cold`, and M1_T2 as the MOSAIC finalist. The warm noninferiority margin was 10% of `C_seq` qualification warm NDCG@20: 0.0030883869550329584.

## Locked results

| Model | Overall NDCG@20 | Controlled cold NDCG@20 | FIT-warm NDCG@20 |
|---|---:|---:|---:|
| B7 `C_seq` | 0.003882 | 0.000000 | 0.027066 |
| B4 `C_cold` | 0.007457 | 0.003996 | 0.009612 |
| M1_T2 | 0.011365 | 0.012936 | 0.003447 |

M1_T2 minus `C_cold` on controlled cold was +0.008940. Its paired user-cluster 95% bootstrap interval was [-0.000194, 0.019205], so superiority failed. Cold item-cluster sensitivity was [-0.001985, 0.024194].

M1_T2 minus `C_seq` on FIT-warm was -0.023619, with paired 95% interval [-0.026874, -0.020417]. The lower bound was below the frozen -0.003088 margin, so noninferiority failed.

The secondary overall difference against `C_seq` was +0.007482, 95% interval [0.006634, 0.008332]. The co-primary design does not permit this secondary positive result to override either failed gate. The terminal decision is **RETAIN_BASELINE**.

## Failure analysis

The locked support analysis sharpened the tradeoff. On zero-FIT-support targets, M1_T2 reached 0.012691 NDCG@20, compared with 0.007097 for `C_cold` and zero for `C_seq`. On FIT support above 20, M1_T2 reached 0.004833, compared with 0.011714 for `C_cold` and 0.036247 for `C_seq`. M1_T2 scored zero in the emerging 1-5 and sparse 6-20 support bins. This pattern is consistent with specialization at the cold boundary rather than broad support-conditioned adaptation.

## Robustness and mechanism evidence

Confirmatory robustness and ablation programs are **INVALID_UNDERSPECIFIED**. The freeze recorded family names but omitted material intensities, scopes, implementation identities, and directional/multiplicity rules. These gaps were discovered after protected access, so p1 does not repair them. The development M0-M7 ladder remains exploratory component evidence. The supported term is **adaptive multimodal fusion**; reliability-aware language is unsupported.

## Replication

MovieLens 32M lacks a native visual modality and has a sharply selected FIT-history evaluation population under the existing split. PixelRec has aligned publisher image features and local text features, plus 3,109 controlled cold items. Neither track had a complete frozen transfer identity, seed budget, access sequence, or external statistics rule before the MicroLens protected outcome. Their terminal replication status is HOLD, and the p1 conclusion is **LOCAL_VALIDITY_ONLY**.

## Systems evidence

The RTX 4090 Laptop GPU processed exact locked full-catalog scoring at 5,927 users/s for B7, 8,469 users/s for B4, and 6,188 users/s for M1_T2. The local FastAPI development adapter completed 100 sequential requests with 4.49 ms median, 5.27 ms p95, 219 requests/s, and zero errors. These are local single-process demonstration results.

`torch.compile` matched eager output within 4.77e-7 and measured 1.044x steady-state speedup, but its 1.81 s cold cost required about 15,396 batches to break even. ANN was not measured because no accuracy/latency operating point was frozen.

## Evidence quality and limitations

The canonical warehouse contains 169 receipts, 772 metric rows, 4 bootstrap intervals, and 13 claims. It validates with three protected execution receipts. Ruff passes and 36 tests pass. Failed jobs, transient CUDA failure, FREEDOM interruption, systems orchestration failures, negative results, and rights holds remain preserved.

Unobserved items are treated as nonrelevant for offline ranking, not verified negatives. First-observed interaction time is a candidate-availability proxy. Content revision times are unavailable. Public observational data provide no production or causal evidence.

## Conclusion

M1_T2 improved the cold point estimate and the secondary overall metric, but the cold confidence interval crossed zero and warm quality fell well beyond the frozen margin. The proposed model is not promoted under p1. A future protocol should directly optimize the cold/warm frontier, observe modality quality in the gate, and preregister exact interventions and external transfer rules.
