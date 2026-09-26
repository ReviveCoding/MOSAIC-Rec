# P2 adversarial audit

1. **Future leakage:** PASS — fixed temporal roles and candidate-time audit.
2. **Split contamination:** PASS — mechanical role audit passed.
3. **Controlled-zero collaborative leakage:** PASS — withheld FIT collaboration audited.
4. **Candidate-time leakage:** PASS — zero recorded violations.
5. **Baseline undertraining:** PASS WITH LIMITATION — equal budgets and independent seeds; absolute scores remain low.
6. **Unequal HPO treatment:** PASS — three trials per neural family and fixed seed policy.
7. **Sampled headline evaluation:** PASS — protected results use exact full catalog.
8. **Best-seed cherry-picking:** PASS — all three independent seeds were ensembled.
9. **License violations:** PASS WITH HOLD — raw-cover derivatives remained held.
10. **Raw-data redistribution:** PASS — large source data remains outside Git.
11. **Invalid content mapping:** PASS — publisher feature alignment audited.
12. **Unobserved-as-negative overclaim:** PASS — ranking metrics do not assert verified negatives.
13. **Gate-weight overinterpretation:** PASS — weights are diagnostics only.
14. **Unsupported reliability language:** PASS — prohibited after 11 directional failures.
15. **Unsupported production language:** PASS — systems evidence is local demonstration only.
16. **Unsupported causal language:** PASS — no causal claim.
17. **Unsupported multi-GPU claim:** PASS — one physical GPU disclosed.
18. **CPU fallback:** PASS — required neural scoring and training used CUDA.
19. **Post-lock mutation:** PASS WITH DEFECT — no model/config mutation; missing ablation runner caused INVALID rather than repair.
20. **Replication overinterpretation:** PASS — transfer is HOLD and no replication claim is made.

## Audit conclusion

The locked primary result is scientifically valid and negative. The ablation family is not valid evidence because its executable pipeline was omitted from the freeze. This defect does not alter the primary result or the permanent p2 decision.
