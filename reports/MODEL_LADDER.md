# MOSAIC development ladder: MicroLens-100K

Protocol `mosaic-rec-2026-09-20-p1`, DEVELOPMENT. These are single-seed, three-epoch CUDA pilots with exact available-catalog top-20 validation scoring. Each row is derived from its `evidence/microlens_m*_pilot.json` receipt. There is no equal-budget HPO, independent-seed qualification, frozen comparator, or protected outcome access.

| Variant | All NDCG@20 | Controlled cold NDCG@20 (n=918) | FIT-warm NDCG@20 (n=20,047) |
|---|---:|---:|---:|
| M0 naïve branch sum | 0.012708 | 0.000000 | 0.026404 |
| M1 fixed fusion | 0.019197 | 0.030476 | 0.005828 |
| M2 learned global fusion | 0.018291 | 0.030245 | 0.000000 |
| M3 support gate | 0.015975 | 0.027928 | 0.000000 |
| M4 behavioral dropout | 0.015313 | 0.035573 | 0.001205 |
| M5 modality dropout | 0.016732 | 0.005140 | 0.029227 |
| M6 consistency signal | 0.017389 | 0.005051 | 0.030447 |
| M7 monotonic gate challenger | 0.015415 | 0.001485 | 0.030546 |

The best cold and warm pilot values occur in different variants. The M2/M3 FIT-warm zeros and the sharp M4-to-M5 reversal warrant evaluator, score-distribution, and seed-stability checks before model selection. The `all` slice includes many naturally cold, non-controlled targets, so it is not a substitute for either co-primary gate.

Gate coefficients are relative branch weights, not calibrated reliability probabilities. M0's original diagnostic gate fields are invalid because M0 actually uses an unweighted sum; its ranking metrics remain usable as pilots. Untimestamped content revisions prevent a strong historical content-leakage exclusion. No reliability or production claim follows from this table.
