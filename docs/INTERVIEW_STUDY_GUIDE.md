# Interview study guide

## Problem and design

- **Question:** Can an adaptive image/text/metadata/behavior fusion model improve controlled inductive cold-item NDCG@20 while remaining noninferior on warm items?
- **Why temporal roles?** FIT-only inputs precede VALIDATION, QUALIFICATION, and LOCKED_FINAL outcomes. The separate protected roles limit tuning on final outcomes.
- **Why controlled cold?** The selected item cohort has no model-visible FIT collaborative support, while permitted content remains available.
- **What is candidate availability?** First observed interaction time is a proxy for observed availability. It is not a verified product launch time.

## Current findings

- M0-M7 are development pilots. M4 led the cold slice and M7 the warm slice; no finalist or promotion is selected.
- A sequence-only ID model cannot be the sole cold comparator; a qualified inductive-capable comparator is required.
- The official PixelRec publisher image vectors cover all track item IDs; raw covers remain quota-blocked.
- Canonical receipts validate with zero locked execution. The post-reboot audit kept the same unsealed protocol because checked artifacts were intact.

## Questions to answer before a scientific claim

1. Were baseline models trained, tuned, and evaluated under common FIT and candidate semantics?
2. Did multiple seeds reproduce the cold/warm tradeoff?
3. Were finalist, comparators, margin, and gates frozen before protected outcome access?
4. Does exact full-catalog locked scoring satisfy both co-primary gates with paired uncertainty?
5. Does the fusion gate respond directionally to controlled signal degradation?
6. Can MovieLens behavioral and PixelRec multimodal tracks support the same scoped finding?

Do not describe public offline metrics as watch-time or production impact.
