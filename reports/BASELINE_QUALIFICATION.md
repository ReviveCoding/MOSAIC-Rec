# Baseline qualification ledger

## Terminal update, 2026-09-23

E00 and E01 completed. Qualification froze B7_BERT4REC_T3 as `C_seq` and B4_TWO_TOWER_T3 as `C_cold`, both as three-seed score ensembles. The common-slice replays matched within 1e-6 and checkpoint hashes were unchanged. Other baseline families remain explicitly unqualified rather than serving as weak comparators. See `reports/FINAL_STATUS.md` for locked results.

## Current status, 2026-09-21

The one-seed, three-learning-rate, 30-epoch CUDA HPO grid is in progress under `configs/hpo_development_p1.json`. B2 BPR-MF, B3 LightGCN, B4 content Two-Tower, B5 corrected GRU4Rec, and B8 VBPR have all three terminal trial receipts. B6 corrected SASRec has all three terminal trial receipts; B7 BERT4Rec now has three terminal CUDA trials; B9 BM3 has three terminal CUDA trials; B10 FREEDOM has three terminal CUDA trials; M1/M4/M6/M7 remain queued. Completed development grid heads by overall NDCG@20 are B2 .008909, B3 .009007, B4 .010669, B5 .014448, B6 .006839, B7 .024235, B8 .016933, B9 .012045, and B10 .012046. The strongest observed controlled-cold NDCG@20 in the B4 and B8 grids is .004774 and .000363, respectively; ID-only models are not cold comparators. These are validation observations; no neural baseline is yet qualified by parity and independent seeds, and neither comparator is frozen. FREEDOM attempt 2 passed a distinct one-epoch CUDA runtime pilot with zero controlled-cold score; attempt 1 remains INTERRUPTED. The corrected BPR, LightGCN, SASRec, GRU4Rec, and local BERT4Rec pilots all completed before HPO. See the canonical HPO receipts and `reports/PROGRESS.md` for current trial status.

## Historical pilot ledger before resumed HPO

All numbers below are **VALIDATION development pilots** under protocol `mosaic-rec-2026-09-20-p1`. No baseline has passed equal-budget HPO, independent-seed qualification and common-protocol reference parity. Therefore neither `C_seq` nor `C_cold` is frozen.

| Family | MicroLens status | Cold capability | Qualification gap |
|---|---|---|---|
| B0 global popularity | Exact FIT-only pilot | No controlled-cold hits | Descriptive comparator only |
| B1 recent popularity | Declared 7/30/90-day FIT-only validation search selected 7 days; overall NDCG@20 0.006341 | No controlled-cold hits | Deterministic development selection; no locked result |
| B2 BPR-MF | Earlier two-epoch CUDA pilot predates FIT-hidden cohort; a separate common FIT-visible 10-epoch CUDA pilot is declared with event-time-safe negative sampling and a known-answer leakage test | Learned IDs lack FIT-hidden support | Corrected common pilot execution, HPO and seed parity pending; old score is not comparable |
| B3 LightGCN | Two-layer CUDA pilot | Zero controlled-cold hits | HPO, seeds and parity pending |
| B4 Two-Tower content | CUDA pilot | Content can score FIT-hidden items | HPO, seeds and reference parity pending |
| B5 GRU4Rec | CUDA pilot used a source-wide negative pool with possible FIT-hidden cold negative updates; future training code now guards the cold cohort | ID-only; historical cold evidence unqualified | Corrected common pilot, HPO, seeds and parity pending |
| B6 SASRec | CUDA pilot had the same possible FIT-hidden cold negative updates; isolated RecBole runtime smoke uses a different internal split | ID-only; historical cold evidence unqualified | Corrected common pilot, HPO, seeds and parity pending |
| B7 BERT4Rec | Isolated RecBole CUDA runtime smoke; local common-protocol Cloze pilot code and CPU cold-gradient test prepared, not trained | ID-only cold limitation | Local CUDA pilot, HPO, reference/local parity and seeds pending |
| B8 VBPR | CUDA pilot | Content can score FIT-hidden items | HPO, seeds and parity pending |
| B9 BM3 | One-epoch MMRec architecture CUDA/common-validation pilot; overall NDCG@20 0.002922, controlled cold 0, FIT-warm 0.006078 | Reference inference scores graph/ID embeddings | Undertrained pilot only; HPO and seed parity pending; not sole cold comparator |
| B10 FREEDOM | Attempt 1 began on CUDA, wrote a hashed item graph cache, then exited abruptly without metric/checkpoint; distinct attempt 2 is prepared | Content item graph may support FIT-hidden items | Runtime failure cause unproven; CUDA retry, HPO and parity pending |

The 7-day selected setting also measured PixelRec overall NDCG@20 0.002820 and MovieLens overall 0.052523. The MovieLens figure applies to 6,465 users with FIT history; 15,879 validation users without FIT history were excluded. The declared search and all nine receipts are in `configs/baseline_b1_search_p1.json` and `evidence/b1_search_selection_p1.json`. These results cannot establish a MOSAIC promotion or any production impact.

`configs/sequence_coldguard_pilot_p1.json` and `scripts/run_corrected_sequence_pilots.sh` declare distinct SASRec/GRU4Rec development reruns with FIT-only event-time negative eligibility, full past-history exclusion, cold-ID assertion, original seed, and separate receipt/checkpoint paths. They have not run while unrelated GPU work is active. Even if they pass, they remain one-seed pilots until HPO, parity and seed qualification close.

Receipt identities are in `evidence/`; the canonical `metrics` table is the source for numerical reporting. Every baseline failure or underqualification remains visible and must not be used as a weak comparator by default.


Convergence watch (2026-09-21, DEVELOPMENT): BERT4Rec lr .003 FIT Cloze loss decreased from 8.59779 at epoch 1 to 6.46926 at epoch 20 and 6.24185 at epoch 30. This fixed-budget HPO result may still be undertrained; do not qualify C_seq from its one-seed grid alone. Complete parity and independent-seed/convergence review before freeze. GRU4Rec lr .001 FIT BPR loss fell from .39264 to .00130 at epoch 20 and .000319 at epoch 30; SASRec lr .001 from .41627 to .04994 and .01281. Training loss is diagnostic, not a protected outcome or a substitute for validation.




FREEDOM convergence watch (2026-09-21, DEVELOPMENT): at lr .0003 FIT loss declined .69196 -> .20635 (epoch 20) -> .13971 (epoch 30); at lr .001, .67070 -> .03955 -> .01491. Its one-seed fixed-budget HPO grid is not convergence or reference/common-parity qualification. Preserve the measured small controlled-cold score at lr .001 (.000248 NDCG@20) without claiming robust cold support.

