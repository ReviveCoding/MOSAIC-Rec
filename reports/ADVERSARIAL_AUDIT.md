# Adversarial scientific audit — development checkpoint

Protocol `mosaic-rec-2026-09-20-p1` remains in DEVELOPMENT. This is a live audit, not K04 closure. The canonical warehouse currently validates 55 receipts and zero locked-execution receipts. A05 mechanical leakage checks passed, but outcome-dependent gates below remain open.

| Risk | Current evidence or issue | Disposition |
|---|---|---|
| Future interaction leakage | Four roles and FIT-visible training artifacts are frozen; source-wide first-observed time is used only as an evaluation availability proxy. A pre-run review corrected the queued BPR sampler so its negative pool and user exclusions use only information available by each FIT event. | Mechanical pass; candidate-time interpretation limited; corrected BPR pilot pending |
| Split contamination | Three-track A06 audits passed after the reboot integrity audit. | PASS mechanical |
| Controlled-cold collaborative leakage | MicroLens 772 and PixelRec 3,109 selected item IDs have positive FIT collaboration hidden. A code review found the historical SASRec/GRU4Rec pilot negative sampler used source-wide item first-observed times, allowing possible negative updates on FIT-hidden cold IDs. Future sequence training now uses FIT_VISIBLE items only and asserts cold exclusion. | Historical sequence cold evidence unqualified; corrected code not yet rerun |
| Candidate-time leakage | First observed interaction is not product launch. Full-source first-observed timestamps determine available candidates at an event. | Proxy limitation disclosed; further sensitivity pending |
| Baseline undertraining | BM3 has one epoch; BERT4Rec only a reference runtime smoke; FREEDOM attempt 1 interrupted; fair HPO absent. | OPEN; no comparator freeze |
| Unequal HPO | No equal-budget neural search has completed. | OPEN; no promotion claim |
| Sampled evaluation headline | Development and BM3 pilots use exact available-catalog scoring. No locked headline exists. | Pending locked audit |
| Best-seed cherry picking | M0-M7 development ladder is single-seed; additional seeds declared but not run. | OPEN |
| License violations | Raw PixelRec covers quota-blocked; MicroLens raw-cover transformation rights unclear; publisher features stay local. | HOLD affected branches |
| Raw-data redistribution | Raw datasets and feature vectors stay in WSL scratch and are ignored by Git. | PASS current workspace audit |
| Invalid content mapping | MicroLens publisher arrays and PixelRec publisher image vectors passed full item-ID mapping/finiteness checks. | PASS measured sources |
| Unobserved-as-negative overclaim | BPR-style training samples unobserved FIT items; no exposure or true-negative labels are known. | Wording constraint |
| Gate-weight overinterpretation | M0 diagnostic softmax did not control its sum; original receipt preserved and marked invalid for interpretation. | PASS correction; later diagnostics pending |
| Reliability language | Controlled intervention and directional gate-response tests have not run. The current gate observes support but no image-quality signal, so a content mismatch at fixed support cannot induce learned gate response. | UNSUPPORTED |
| Production language | Local FastAPI adapter is a development demonstration; no throughput/latency study or deployment. | UNSUPPORTED |
| Causal language | Public interactions are observational; no randomized exposure. | UNSUPPORTED |
| Multi-GPU claims | One RTX GPU is present. | UNSUPPORTED |
| CPU fallback of neural stages | CUDA requalified after reboot; completed neural pilots report RTX device. BM3 and FREEDOM attempt 1 used CUDA. | PASS observed jobs; later jobs pending |
| Post-lock scientific mutation | No locked execution receipt exists. | Not applicable yet; enforce at freeze |
| External replication interpretation | MovieLens and PixelRec final model replication has not run. | OPEN |

K04 can close only after comparator/seed parity, frozen locked scoring, robustness, ablations, replication, systems, and canonical claim validation. Unfavorable model results must remain in the warehouse.
