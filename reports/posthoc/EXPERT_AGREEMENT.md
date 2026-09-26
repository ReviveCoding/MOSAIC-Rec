# Expert Agreement

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

Per-query win/tie/loss, the four-cell success matrix, and top-k overlap are persisted for overall, cold, warm, support, and item-age slices.

| Success matrix cell | N | Fraction | R4 NDCG | Mean routing weight |
| --- | --- | --- | --- | --- |
| both_experts_fail | 25287 | 0.996846 | 0.001201 | 0.196309 |
| c_seq_only_succeeds | 9 | 0.000355 | 0.000000 | 0.945168 |
| c_cold_only_succeeds | 71 | 0.002799 | 0.231677 | 0.146292 |
| both_succeed | 0 | 0.000000 | 0.000000 | 0.000000 |

Top-20 Jaccard, top-1 agreement, and intersection sizes quantify list agreement without treating unobserved interactions as verified negatives.
