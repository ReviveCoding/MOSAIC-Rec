# Baseline qualification ledger

All numbers below are **VALIDATION development pilots** under protocol `mosaic-rec-2026-09-20-p1`. No baseline has passed equal-budget HPO, independent-seed qualification and common-protocol reference parity. Therefore neither `C_seq` nor `C_cold` is frozen.

| Family | MicroLens status | Cold capability | Qualification gap |
|---|---|---|---|
| B0 global popularity | Exact FIT-only pilot | No controlled-cold hits | Descriptive comparator only |
| B1 recent popularity | Declared 7/30/90-day FIT-only validation search selected 7 days; overall NDCG@20 0.006341 | No controlled-cold hits | Deterministic development selection; no locked result |
| B2 BPR-MF | Two-epoch CUDA pilot | Learned IDs lack FIT-hidden support | Earlier pilot predates the FIT-hidden cohort; not common-protocol comparable |
| B3 LightGCN | Two-layer CUDA pilot | Zero controlled-cold hits | HPO, seeds and parity pending |
| B4 Two-Tower content | CUDA pilot | Content can score FIT-hidden items | HPO, seeds and reference parity pending |
| B5 GRU4Rec | CUDA pilot | ID-only cold limitation | HPO, seeds and parity pending |
| B6 SASRec | CUDA pilot; isolated RecBole runtime smoke | ID-only cold limitation | RecBole uses internal FIT-only split; no common-protocol parity |
| B7 BERT4Rec | Isolated RecBole CUDA runtime smoke | ID-only cold limitation | Local common-protocol evaluator/training, HPO and parity pending |
| B8 VBPR | CUDA pilot | Content can score FIT-hidden items | HPO, seeds and parity pending |
| B9 BM3 | Official reference code and hash-aligned inputs inspected; local adapter syntax checked | Reference inference scores graph/ID embeddings | CUDA runtime/common-validation pilot, HPO and seed parity pending; not sole cold comparator |
| B10 FREEDOM | Official reference code inspected; padded-item-safe MicroLens adapter syntax checked | Content item graph may support FIT-hidden items | Declared CUDA runtime/common-validation pilot, HPO and parity pending |

The 7-day selected setting also measured PixelRec overall NDCG@20 0.002820 and MovieLens overall 0.052523. The MovieLens figure applies to 6,465 users with FIT history; 15,879 validation users without FIT history were excluded. The declared search and all nine receipts are in `configs/baseline_b1_search_p1.json` and `evidence/b1_search_selection_p1.json`. These results cannot establish a MOSAIC promotion or any production impact.

Receipt identities are in `evidence/`; the canonical `metrics` table is the source for numerical reporting. Every baseline failure or underqualification remains visible and must not be used as a weak comparator by default.
