# Router Root Cause

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

The persisted value is reported as **sequence routing weight**, not as a probability. Overall routing-side agreement with the empirically stronger expert was 0.002720; disagreement was 0.000434; neutral routing was 0.000000. Experts tied on 0.996846 of queries. Conditional on the 80 queries where one expert was stronger and routing chose a side, agreement was 0.862500.

Spearman association between sequence routing weight and per-query C_seq minus C_cold NDCG was 0.018977. This is an association measured after outcomes and is not a causal or reliability claim.

Router-weight deciles and every preregistered covariate grouping are in the machine-readable tables.
