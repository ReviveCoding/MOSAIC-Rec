# Final adversarial scientific audit

Protocol: `mosaic-rec-2026-09-20-p1`. Terminal decision: `RETAIN_BASELINE`.

| Risk | Evidence | Final disposition |
|---|---|---|
| 1. Future leakage | Four temporal roles; time-safe FIT samplers; first-observed time used only as disclosed availability proxy | PASS with proxy limitation |
| 2. Split contamination | Hash/schema and cross-role mechanical audits passed on all three tracks | PASS |
| 3. Cold collaborative leakage | MicroLens 772 and PixelRec 3,109 FIT-hidden items; corrected neural pools excluded cold IDs | PASS for qualified runs; historical pilots remain unqualified |
| 4. Candidate-time leakage | Candidate availability uses first observed interaction, not launch time | LIMITATION, disclosed |
| 5. Baseline undertraining | Equal fixed HPO grids, independent E01 seeds, reference smokes, separate B7/B4 comparators | PASS for frozen comparators; other families explicitly unqualified |
| 6. Unequal HPO | Comparable frozen candidates used declared three-trial budgets; all failures retained | PASS within declared grid |
| 7. Sampled headline evaluation | Qualification and locked results use exact full catalog | PASS |
| 8. Best-seed cherry picking | Frozen three-seed score ensembles | PASS |
| 9. License violations | Source/terms audit; raw-cover derivative work held | PASS current use; HOLD derivative work |
| 10. Raw-data redistribution | Raw data/features stay in ignored WSL scratch | PASS |
| 11. Invalid content mapping | Publisher arrays/features passed full item alignment and finiteness checks | PASS |
| 12. Unobserved-as-negative overclaim | Reports describe offline nonrelevance, not true negatives | PASS wording constraint |
| 13. Gate-weight overinterpretation | M0 invalid diagnostic weights excluded; coefficients not called probabilities | PASS |
| 14. Unsupported reliability language | Gxx invalid due incomplete frozen interventions | PASS by prohibiting claim |
| 15. Unsupported production language | FastAPI evidence labeled local development demonstration | PASS |
| 16. Unsupported causal language | No causal or engagement claim in ledger | PASS |
| 17. Unsupported multi-GPU claims | Single-GPU evidence only | PASS |
| 18. CPU fallback | Neural receipts identify RTX CUDA; no silent long CPU fallback | PASS |
| 19. Post-lock mutation | Original and amended pre-access freezes preserved; post-lock G/H gaps left invalid | PASS |
| 20. Replication overinterpretation | No external neural result; status `LOCAL_VALIDITY_ONLY` | PASS by holding claim |

## Defects found and handling

- A pre-access E05 query label defect was amended before any protected receipt and re-guarded under a new hash.
- The first locked command failed its stage-token guard before reading protected data; the failure is preserved.
- Robustness and ablation definitions were materially incomplete at freeze. They remain INVALID rather than being repaired after outcomes.
- External transfer rules were not frozen before primary protected access. Replication remains HOLD.
- Two systems attempts failed through CUDA-process isolation and lock reentrancy. Both failures are preserved; the corrected serial measurement succeeded without scientific changes.

The audit found no evidence requiring invalidation of the exact locked scoring or primary bootstrap inference. It does prevent reliability, confirmatory mechanism, external replication, production, causal, and multi-GPU claims.
