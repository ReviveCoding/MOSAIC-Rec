# Post-hoc p1 diagnostic report

**POST-HOC EXPLORATORY - NOT A P1 CONFIRMATORY CLAIM**

This report reads immutable p1 per-user outputs and FIT-visible context only. It performs no model inference, creates no canonical evidence, and cannot change the permanent `RETAIN_BASELINE` decision.

## Scope and source integrity

- Analysis type: `POSTHOC_EXPLORATORY`.
- Confirmatory: `false`.
- Bootstrap: 2,000 deterministic paired user resamples, seed 20260923; item-cluster sensitivity for the controlled-cold slice.
- Locked users analyzed: 38,880; qualification users: 40,296.
- Target items represented in locked analysis: 5,421.
- Every source was hashed before and after analysis with no change.

## 1. Where does MOSAIC help?

Against B4 on the controlled-cold cohort, the existing locked rows show an exploratory mean NDCG@20 delta of 0.008940, with paired post-hoc interval [0.000188, 0.018324]. This reproduces the direction of the closed p1 observation but does not upgrade it to superiority.

The strongest reportable head/torso/tail segment versus B4 was `zero` (n=33,303, delta=0.005594). The strongest item-age segment was `8-30d` (n=9,303, delta=0.005097). These are discovered post hoc.

## 2. Where does it hurt?

Against B7 on the frozen warm slice, the exploratory mean NDCG@20 delta is -0.023619, interval [-0.026920, -0.020212]. The worst reportable warm user segment among the requested families was `base_x_diversity_quantile:warm_fit|Q1` (n=1,749, delta=-0.038332).

The secondary overall delta versus B7 remains positive at 0.007482, which illustrates why aggregate performance masked the warm failure.

## 3. Is the cold advantage concentrated?

| FIT support | n | M1-B4 NDCG delta | Exploratory 95% interval |
|---|---:|---:|---:|
| 0 | 33,303 | 0.005594 | [0.004733, 0.006394] |
| 1-5 | 462 | -0.004565 | [-0.008493, -0.001454] |
| 6-20 | 1,137 | -0.004308 | [-0.007481, -0.001916] |
| 21-50 | 1,820 | -0.005619 | [-0.008227, -0.003370] |
| 51-100 | 1,502 | -0.005567 | [-0.010767, -0.000592] |
| 101+ | 656 | -0.013387 | [-0.022028, -0.004344] |

The detailed machine-readable tables also segment by FIT interaction count, unique FIT users, fixed popularity quintiles/deciles, head/torso/tail, item age, and age-by-support cells with n>=200. Concentration by popularity or age is descriptive and noncausal.

## 4. What explains the warm collapse?

M1 warm NDCG@20 changed from 0.014701 in QUALIFICATION to 0.003447 in LOCKED_FINAL, a change of -0.011253. B7 changed from 0.030884 to 0.027066, a change of -0.003818.

The largest absolute composition shift among the audited bins was `recency_bin:8-30d` at 0.1939. Because M1's warm decline was materially larger than B7's under the same role change, the descriptive evidence is more consistent with an M1-specific sensitivity interacting with context than with a population-only explanation. This is not causal attribution.

## 5. Did qualification-to-locked drift contribute?

Composition drift exists and is quantified in `population_drift.parquet` using absolute share shift, Jensen-Shannon distance, and one-dimensional Wasserstein distance. The model-specific metric changes show that common population change does not fully explain the divergence, but the immutable artifacts cannot isolate a causal mechanism.

## 6. Which effects are stable versus heterogeneous?

The E01 table preserves mean, sample SD, coefficient of variation where meaningful, and min/max for every frozen candidate and slice. It is DEVELOPMENT evidence only. M1_T2 showed nonzero seed variation, and the support, popularity, age, and user slices show substantial heterogeneity in locked deltas. No candidate is reselected.

Warm-item user effects were negative across the requested reportable cohort families. The least and most negative cells were:

| User family | Least negative warm cell | Delta | Most negative warm cell | Delta |
|---|---|---:|---|---:|
| history_bin | 1-2 (n=1,612) | -0.015847 | 6-10 (n=848) | -0.031538 |
| recency_bin | 31-90d (n=2,443) | -0.020305 | 8-30d (n=2,993) | -0.027069 |
| niche_quantile | Q1 (n=1,547) | -0.015232 | Q4 (n=1,204) | -0.037236 |
| tail_share_quantile | Q3 (n=1,646) | -0.015329 | Q2 (n=3,931) | -0.027090 |
| diversity_quantile | Q3 (n=1,872) | -0.011824 | Q1 (n=1,749) | -0.038332 |

Sparse-history, inactive/returning, niche/long-tail preference, and diversity segments therefore change the magnitude of the warm loss rather than revealing a reportable warm cohort where M1 beats B7. These contrasts are hypothesis-generating and use FIT-visible user features only.

## 7. Content-quality diagnostics

Publisher image and text vectors were present for all catalog rows. Title presence, tag-summary presence, text-length bins, and publisher feature-norm outliers were analyzed. Multi-tag count is `NOT_AVAILABLE` because `tags_to_summary.csv` supplies one summary category rather than a valid multi-tag set. No raw-cover feature was created.

## 8. Case studies

`case_studies.csv` contains four deterministic qualitative illustrations: the largest controlled-cold M1 rescue, largest warm regression, median cold win, and median warm loss. They use permitted title/tag summaries and do not support general claims.

## 9. Metrics not computable

The immutable per-user artifacts contain target ranks and per-user metrics but no top-k item lists. Catalog coverage@20, recommendation popularity, long-tail exposure, exposure concentration, intra-list diversity, novelty, personalization, and list overlap are marked `DEFER_TO_P2_REQUIRES_NEW_PREDECLARED_SCORING`. They were not approximated and protected scoring was not rerun.

## 10. What p2 should preregister

- The exact support, popularity, age, and user cohort hypotheses selected from this exploratory report.
- Fixed slice boundaries, minimum sample rules, bootstrap unit, draw count, and multiplicity policy.
- A gate input that can observe content quality if reliability behavior is tested.
- Complete intervention intensities and ablation implementations.
- External transfer identities and access order before PixelRec outcomes.
- Top-k retention if coverage, exposure, diversity, novelty, or personalization will be evaluated.

## Guardrail

Nothing in this report amends p1. The authoritative decision remains `RETAIN_BASELINE`.
