# Exposure and Catalog Health

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

## Overall catalog metrics

| Model | Coverage@20 | Avg popularity | Tail % | Gini | HHI | Entropy | Novelty | ILD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_seq | 0.003439 | 4.176156 | 0.000000 | 0.999560 | 0.025877 | 3.887577 | 13.477413 | 0.632214 |
| C_cold | 0.347915 | 1.557811 | 0.094091 | 0.922728 | 0.000286 | 8.918861 | 17.254887 | 0.410711 |
| R4 | 0.166234 | 0.404723 | 0.000000 | 0.968100 | 0.000714 | 8.041808 | 18.918441 | 0.451266 |

Relative to C_seq, R4 coverage changed by 0.162795, tail exposure by 0.000000, and Gini by -0.031460. Relative to C_cold, the same deltas were -0.181681, -0.094091, and 0.045372. These are offline catalog-health descriptions, not fairness or engagement claims.
