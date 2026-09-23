# Amazon Applied Scientist positioning

## Core positioning

MOSAIC-Rec is best presented as an end-to-end recommendation research project in which scientific judgment mattered as much as model implementation. It connects a concrete personalization problem to temporal evaluation, multimodal modeling, statistical gates, GPU execution, and an auditable decision.

## Capability map

| Area | Project evidence | Safe positioning |
|---|---|---|
| Recommendation and personalization | Sequential ranking, exact full-catalog evaluation, distinct warm and cold requirements | Designed a temporal recommendation study around user history and item support |
| Cold start | Controlled FIT-hidden item cohort and content-capable comparator | Evaluated inductive cold-item behavior without exposing FIT collaboration |
| Multimodal modeling | Collaborative, text, image, and metadata branches with adaptive fusion variants | Built and compared multimodal sequence-fusion architectures |
| Experimentation | DEVELOPMENT, QUALIFICATION, and LOCKED_FINAL separation with frozen selection rules | Prevented tuning and model selection on protected outcomes |
| Statistical inference | Paired user-cluster bootstrap, item-cluster sensitivity, and a frozen noninferiority margin | Used uncertainty and decision thresholds instead of point estimates alone |
| ML systems | CUDA training and exact scoring, cached features, a GPU lock, canonical receipts, and a FastAPI demo | Built a reproducible single-GPU research and demonstration stack |
| Research rigor | Preserved failures, negative results, invalid stages, and stronger prohibited claims | Closed with `RETAIN_BASELINE` when the finalist failed both co-primary gates `[terminal_decision]` |

## Evidence-backed impact story

The strongest cold point estimate was encouraging: M1_T2 exceeded the frozen cold comparator by `0.008940` NDCG@20. The paired interval crossed zero, so the project did not claim superiority. `[locked_cold_point_improvement]`

The same finalist was `0.023619` below the sequence comparator on warm items and failed the noninferiority gate. `[locked_warm_noninferiority_failed]`

Although the overall secondary comparison favored M1_T2 by `0.007482`, the frozen decision rule required success on both primary dimensions. `[locked_overall_secondary]` The resulting `RETAIN_BASELINE` decision demonstrates disciplined judgment under mixed evidence. `[terminal_decision]`

## Themes to emphasize in an interview

### Customer-oriented problem formulation

Frame cold start and established-item quality as separate user-facing requirements. Explain that a recommender that helps only one cohort can still be unsuitable for broad use.

### Scientific ownership

Describe the complete chain: source and license checks, temporal roles, leakage guards, baseline qualification, model selection, protected evaluation, inference, claim audit, and closure.

### High standards for evidence

Explain why the project used two comparators, exact full-catalog ranking, paired inference, and a noninferiority gate. Emphasize that the positive aggregate result did not change the predeclared decision rule.

### Practical systems judgment

Discuss running one heavy CUDA workload at a time, maximizing batch throughput, preserving checkpoints and receipts, recovering from a reboot, and separating local systems measurements from scientific performance. The local API's `4.49 ms` median result is demonstration evidence only. `[systems_local_measurement]`

### Learning from failure

The robustness and ablation program was not presented as successful after the audit found incomplete pre-lock definitions. The next protocol would freeze exact interventions and implementations before protected access.

## Claims to avoid

- Do not say MOSAIC was promoted or established cold superiority.
- Do not say warm quality was retained.
- Do not describe the gate as reliability-aware or calibrated.
- Do not claim external replication, causal lift, production impact, production readiness, or multi-GPU scaling.
- Do not turn `RETAIN_BASELINE` into a universal claim that multimodal recommendation is ineffective.

## Suggested positioning statement

> I built an auditable multimodal sequential recommendation study focused on cold-start quality and warm-item retention. I qualified separate comparators, froze three-seed selection and statistical gates before protected evaluation, and ran exact CUDA full-catalog scoring. The model improved the cold point estimate and the overall secondary metric, but uncertainty crossed the cold superiority boundary and warm noninferiority failed, so I retained the baseline. The project shows how I connect modeling, inference, systems execution, and disciplined scientific decisions without overstating offline evidence.
