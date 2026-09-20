# Baseline qualification ledger

All numbers below are **VALIDATION development pilots** under protocol `mosaic-rec-2026-09-20-p1`. No baseline has passed equal-budget HPO, independent-seed qualification and common-protocol reference parity. Therefore neither `C_seq` nor `C_cold` is frozen.

| Family | MicroLens status | Cold capability | Qualification gap |
|---|---|---|---|
| B0 global popularity | Exact FIT-only pilot | No controlled-cold hits | Descriptive comparator only |
| B1 recent popularity | Declared 7/30/90-day FIT-only validation search selected 7 days; overall NDCG@20 0.006341 | No controlled-cold hits | Deterministic development selection; no locked result |
| B2 BPR-MF | Earlier two-epoch CUDA pilot predates FIT-hidden cohort; a separate common FIT-visible 10-epoch CUDA pilot is declared with event-time-safe negative sampling and a known-answer leakage test | Learned IDs lack FIT-hidden support | Corrected common pilot execution, HPO and seed parity pending; old score is not comparable |
| B3 LightGCN | Two-layer CUDA pilot | Zero controlled-cold hits | HPO, seeds and parity pending |
| B4 Two-Tower content | CUDA pilot | Content can score FIT-hidden items | HPO, seeds and reference parity pending |
| B5 GRU4Rec | CUDA pilot | ID-only cold limitation | HPO, seeds and parity pending |
| B6 SASRec | CUDA pilot; isolated RecBole runtime smoke | ID-only cold limitation | RecBole uses internal FIT-only split; no common-protocol parity |
| B7 BERT4Rec | Isolated RecBole CUDA runtime smoke | ID-only cold limitation | Local common-protocol evaluator/training, HPO and parity pending |
| B8 VBPR | CUDA pilot | Content can score FIT-hidden items | HPO, seeds and parity pending |
| B9 BM3 | One-epoch MMRec architecture CUDA/common-validation pilot; overall NDCG@20 0.002922, controlled cold 0, FIT-warm 0.006078 | Reference inference scores graph/ID embeddings | Undertrained pilot only; HPO and seed parity pending; not sole cold comparator |
| B10 FREEDOM | Attempt 1 began on CUDA, wrote a hashed item graph cache, then exited abruptly without metric/checkpoint; distinct attempt 2 is prepared | Content item graph may support FIT-hidden items | Runtime failure cause unproven; CUDA retry, HPO and parity pending |

The 7-day selected setting also measured PixelRec overall NDCG@20 0.002820 and MovieLens overall 0.052523. The MovieLens figure applies to 6,465 users with FIT history; 15,879 validation users without FIT history were excluded. The declared search and all nine receipts are in `configs/baseline_b1_search_p1.json` and `evidence/b1_search_selection_p1.json`. These results cannot establish a MOSAIC promotion or any production impact.

Receipt identities are in `evidence/`; the canonical `metrics` table is the source for numerical reporting. Every baseline failure or underqualification remains visible and must not be used as a weak comparator by default.
