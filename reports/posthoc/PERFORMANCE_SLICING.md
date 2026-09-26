# Performance Slicing

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

## Overall model quality

| Model | NDCG@20 | Recall@20 | MRR@20 | Hit@20 | Precision@20 |
| --- | --- | --- | --- | --- | --- |
| C_seq | 0.000111 | 0.000355 | 0.000046 | 0.000355 | 0.000018 |
| C_cold | 0.001008 | 0.002799 | 0.000541 | 0.002799 | 0.000140 |
| R4 | 0.001846 | 0.005204 | 0.000952 | 0.005204 | 0.000260 |

The strongest non-small descriptive slice delta was `popularity_tier / torso` for `R4_minus_C_seq`: 0.003296 (exploratory 95% interval 0.002447, 0.004216; N=8817). The largest descriptive risk was `support / >50` for `R4_minus_C_seq`: -0.004394 (N=520). These rankings are diagnostics and do not change model selection.

Crossed slices were emitted only for N >= 200. Exact quartile edges are persisted in `analysis/posthoc/quartile_edges.json`.
