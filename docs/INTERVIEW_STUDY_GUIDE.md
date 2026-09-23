# Interview study guide

## Design

- Four temporal roles separated development, qualification, and protected final evidence.
- Controlled cold items had content but no model-visible FIT collaboration.
- Two comparators represented different requirements: sequence quality and inductive cold capability.
- Qualification used frozen three-seed score ensembles and a rank-balancing finalist rule.

## Result

- M1_T2 cold NDCG@20 was 0.012936 versus 0.003996 for the cold comparator, but the difference interval crossed zero.
- M1_T2 warm NDCG@20 was 0.003447 versus 0.027066 for BERT4Rec and failed noninferiority.
- The positive overall secondary result did not override the co-primary design.
- The terminal decision was `RETAIN_BASELINE`.

## Lessons

1. Separate cold and warm comparators expose tradeoffs hidden by one aggregate metric.
2. A point improvement is not superiority when paired uncertainty crosses zero.
3. Protected outcomes must not be used to fill gaps in intervention or replication definitions.
4. Gate weights are relative coefficients, not reliability probabilities.
5. Offline public data do not establish production or causal impact.

## Next study

Preregister a new protocol with an explicit cold/warm frontier objective, exact robustness intensities, ablation implementations, and PixelRec transfer rules. Add an observable content-quality signal before testing reliability-aware gating.
