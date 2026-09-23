# MOSAIC-Rec final status

Generated 2026-09-23 under protocol `mosaic-rec-2026-09-20-p1`.

## Terminal decision

**RETAIN_BASELINE.** The frozen MOSAIC finalist, `M1_T2`, failed both co-primary MicroLens LOCKED_FINAL gates. The project is technically closed with an unfavorable scientific result. The result also remains **LOCAL_VALIDITY_ONLY** because no valid frozen external transfer protocol was defined before the primary protected outcome was opened.

## Identity

- Scientific source Git identity: `3506a85` (`Complete post-lock analysis and systems evidence`).
- Scientific state: `CLOSED`.
- Runtime: Ubuntu-22.04 under WSL; isolated environment `/home/bjw-0/mosaic-rec-work/.venv`.
- Device: NVIDIA GeForce RTX 4090 Laptop GPU; required neural work reported CUDA device identities.
- Canonical evidence: 169 receipts, 3 LOCKED_EXECUTION receipts, 772 metric rows, 4 bootstrap intervals, 13 claim rows.
- Validation: canonical warehouse PASS; Ruff PASS; pytest **36 passed**.
- Frozen manifest amendment SHA-256: `99ddc01542078676e9c09809411cdbb68bcae2c8ada19b8498f766f8b5e185c8`.

## Dataset status

| Track | Terminal status | Limitation |
|---|---|---|
| MicroLens-100K | Primary locked study complete | Historical content revision times are unavailable; first-observed interaction time is an availability proxy |
| MovieLens 32M | FIT input and temporal QA PASS; replication HOLD | No frozen transfer model/population/resource protocol; no native visual modality |
| PixelRec50K | Data, raw archive integrity, publisher feature alignment, and FIT input PASS; replication HOLD | Raw-cover derived-feature rights remain held; no frozen external model/statistics protocol |

Mechanical leakage audits passed on all three tracks. Controlled FIT-hidden cohorts contained 772 MicroLens and 3,109 PixelRec items.

## Frozen identities

- Sequence comparator `C_seq`: `B7_BERT4REC_T3`, three-seed score ensemble.
- Cold-capable comparator `C_cold`: `B4_TWO_TOWER_T3`, three-seed score ensemble.
- MOSAIC finalist: `M1_T2`, three-seed score ensemble.
- Warm noninferiority margin: `0.0030883869550329584` NDCG@20.
- Inference: 10,000-draw paired user-cluster bootstrap; cold item-cluster sensitivity.

## Locked primary results

| Model | Overall NDCG@20 | Controlled cold | FIT-warm |
|---|---:|---:|---:|
| `C_seq` B7 | 0.003882 | 0.000000 | 0.027066 |
| `C_cold` B4 | 0.007457 | 0.003996 | 0.009612 |
| `M1_T2` | 0.011365 | 0.012936 | 0.003447 |

Cold difference versus `C_cold` was `+0.008940`; paired 95% interval `[-0.000194, 0.019205]`. The lower bound did not exceed zero, so cold superiority failed.

Warm difference versus `C_seq` was `-0.023619`; paired 95% interval `[-0.026874, -0.020417]`. This was far below the frozen noninferiority threshold of `-0.003088`, so warm noninferiority failed.

The secondary overall difference versus `C_seq` was `+0.007482`, paired 95% interval `[0.006634, 0.008332]`. It does not replace the failed co-primary gates.

## Robustness, ablation, and replication

- Robustness: **INVALID** for confirmatory use. E04/E05 froze family names but omitted material intervention intensities, scopes, and directional statistics. Reliability-aware language is prohibited.
- Ablations: **INVALID** for confirmatory use. Removal implementations, applicability mapping, retraining identities, and multiplicity mapping were not fully frozen. The M0-M7 development ladder remains exploratory evidence.
- Replication: **HOLD / LOCAL_VALIDITY_ONLY**. FIT adapters and QA exist, but no fixed external transfer protocol was frozen before primary outcome access. No external result was tuned until agreement.

## Systems conclusion

- Local text extraction: 3,815 MicroLens and 2,407 PixelRec items/s.
- Exact full-catalog locked scoring: 5,927 users/s for B7, 8,469 users/s for B4, and 6,188 users/s for M1_T2 over 38,880 users.
- Development FastAPI demo: 4.49 ms median and 5.27 ms p95 sequential recommendation latency across 100 requests; zero errors; 219 requests/s in the local single-process test.
- `torch.compile`: numerical max absolute difference `4.77e-7`, 1.044x steady-state speedup, 1.81 s cold cost, about 15,396 batches to break even. Adoption is workload dependent.
- ANN: HOLD because no index family or accuracy/latency operating point was frozen. Exact ranking remains the scientific evaluator.
- Serving remains demonstration evidence and does not establish production readiness.

## Stage dispositions

| Stage family | Status |
|---|---|
| A00-A06 data/provenance/splits | PASS, with declared media rights holds |
| B00-B01 features | PASS for qualified publisher/text features |
| C00-C05 baselines/reference checks | PASS or explicitly unqualified pilot |
| D00-D08 MOSAIC ladder/stability | PASS as DEVELOPMENT evidence |
| E00 HPO | `HPO_GRID_COMPLETE_UNQUALIFIED` |
| E01 seed qualification | `SEED_QUALIFICATION_COMPLETE_UNQUALIFIED` |
| E02-E05 selection/freeze | PASS |
| F00 exact locked scoring | PASS |
| F01 cold gate | FAIL |
| F02 warm gate | FAIL |
| F03-F05 slice/failure analysis | PASS |
| Gxx robustness | INVALID underspecified |
| Hxx ablations | INVALID underspecified |
| I00-I02 replication | HOLD; `LOCAL_VALIDITY_ONLY` |
| J00-J05 systems | PARTIAL PASS; ANN HOLD |
| K00-K05 canonical closure | PASS |

## Supported claims

- Under p1, the M1_T2 cold point estimate exceeded the frozen cold baseline, but superiority was not established.
- Under p1, M1_T2 failed warm noninferiority by a wide margin.
- The secondary overall locked metric favored M1_T2.
- The evidence-supported decision is `RETAIN_BASELINE`.
- The local systems measurements above are reproducible demonstration measurements.

## Unsupported claims

- `CORE_PROMOTE` or `RELIABILITY_PROMOTE`.
- Controlled-cold superiority.
- Warm quality retention.
- Reliability-aware or calibrated gating.
- External replication.
- Production impact, production readiness, causal benefit, or multi-GPU scaling.

## Unresolved blockers and holds

- PixelRec raw-cover transformation and derived-feature rights require explicit source permission.
- A new replication study needs a preregistered transfer model, seed budget, role-access sequence, population, and statistics plan.
- A new robustness/ablation protocol must freeze exact interventions and implementations before protected access.

No user action is required to interpret or reproduce p1. Permission from the relevant data owner is required before raw-cover derivative work.

## Exact reproduction

Run from the Windows repository while keeping scientific Python in Ubuntu-22.04:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec && PYTHONPATH=src /home/bjw-0/mosaic-rec-work/.venv/bin/python -m mosaic_rec.validate_evidence evidence evidence/warehouse.duckdb'
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec && PYTHONPATH=src /home/bjw-0/mosaic-rec-work/.venv/bin/python -m pytest -q'
wsl -d Ubuntu-22.04 -- bash -lc 'cd /mnt/c/Users/bjw-0/Downloads/MOSAIC-Rec && /home/bjw-0/.local/bin/uvx ruff==0.16.8 check src tests scripts'
```

Protected scoring is intentionally not rerunnable under the now-closed guard. Source locked receipts, per-user Parquet hashes, freeze identities, and bootstrap output provide the audit trail. See `docs/REPRODUCTION_RUNBOOK.md` for pre-lock reconstruction.

## Recommended next research step

Open a new preregistered protocol that retains p1 unchanged, targets the observed cold/warm tradeoff, freezes the complete robustness and ablation implementations, and defines PixelRec transfer before accessing external outcomes. Develop a gate with an observable image-quality signal if reliability language remains a goal.
