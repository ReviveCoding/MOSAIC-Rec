# Industry Diagnostic Report

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

## Source boundary

Canonical p2 remains CLOSED with `RETAIN_BASELINE`. All computations read existing locked, robustness, context, feature, and systems artifacts. No protected scoring or training occurred.

## Model quality

| Model | NDCG@20 | Recall@20 | MRR@20 | Hit@20 | Precision@20 |
| --- | --- | --- | --- | --- | --- |
| C_seq | 0.000111 | 0.000355 | 0.000046 | 0.000355 | 0.000018 |
| C_cold | 0.001008 | 0.002799 | 0.000541 | 0.002799 | 0.000140 |
| R4 | 0.001846 | 0.005204 | 0.000952 | 0.005204 | 0.000260 |

## Answers to the diagnostic questions

1. **Where R4 helps:** the strongest qualifying descriptive slice is `popularity_tier / torso` for `R4_minus_C_seq`, delta 0.003296.
2. **Where R4 fails:** the largest slice risk is `support / >50`, delta -0.004394.
3. **Cold gain breadth:** controlled-zero gain is concentrated: 1 positive queries among 295, top-10% positive-gain share 1.000000.
4. **H1 lower bound:** sparse, concentrated nonzero gains left substantial bootstrap mass at zero.
5. **Warm collapse:** R4 was below C_seq in the >50-support slice by -0.004394. Its sequence routing weight was high in this slice, yet the hybrid lost sparse C_seq successes; this points descriptively to score-fusion ranking disturbance rather than failure to weight the sequence side. It is not a causal explanation.
6. **Router choice:** unconditional agreement was 0.002720 because experts tied on 0.996846 of queries. Among informative non-ties, agreement was 0.862500.
7. **Oracle ratio:** R4 was 164.97% of max-per-query expert NDCG. Values above 100% occur because R4 generates a distinct ranking; this is not evidence of exceeding a true oracle.
8. **Main bottleneck:** weak experts overall, with localized routing/fusion errors; both absolute oracle quality and routing regret matter.
9. **Missingness direction:** the predefined conflation hypothesis is `WEAK_ASSOCIATION` from persisted routing deltas; causal mechanism remains unproven.
10. **Catalog health:** R4 vs C_seq changed coverage 0.162795, tail exposure 0.000000, and Gini -0.031460.
11. **Temporal stability:** R4 max-minus-min quartile NDCG variation was 0.001095.
12. **Strongest positive slice:** `popularity_tier / torso` (R4_minus_C_seq).
13. **Largest risk:** `support / >50` (R4_minus_C_seq).
14. **P3 architecture:** constrain zero-support routing, calibrate expert score scales, train explicit expert-selection supervision on validation-only labels, and add warm-preservation objectives before protected access.
15. **Safe new numbers:** only values labeled posthoc exploratory with source commit and population; no number changes the p2 RETAIN_BASELINE decision.
