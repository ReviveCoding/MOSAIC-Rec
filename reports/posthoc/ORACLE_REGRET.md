# Oracle and Routing Regret

> **POST-HOC EXPLORATORY — NOT A P2 CONFIRMATORY CLAIM**

Source protocol: `mosaic-rec-2026-09-23-p2`  
Source commit: `b9bdc1bcb6abf912687ffb304f426107349cd06f`  
Protected scoring rerun: `false`  
Canonical p2 mutated: `false`

The retrospective per-query expert oracle achieved NDCG@20 0.001119; R4 achieved 0.001846; the better single expert achieved 0.001008. R4 captured 164.97% of oracle NDCG. Mean oracle minus R4 regret was -0.000727; R4 exceeded this two-expert oracle on 0.44% of queries because R4 can produce a ranking different from both input experts.

The diagnostic classification is **weak experts overall, with localized routing/fusion errors**. Mean `oracle - R4` is negative because R4 creates a distinct ranking and sometimes exceeds both experts, so this max-expert construction is not a mathematical upper bound on the hybrid. Positive-regret queries still reveal localized routing or fusion failures. This oracle is retrospective and undeployable.
