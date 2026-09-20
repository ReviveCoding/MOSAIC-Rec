# Project narrative (current evidence)

MOSAIC-Rec asks whether temporal recommendation can improve controlled inductive cold-item ranking without giving up warm-item ranking. It uses item image, text, metadata, and FIT-only collaborative support in a sequence model. The primary design separates FIT, VALIDATION, QUALIFICATION, and LOCKED_FINAL roles and removes FIT interactions for a fixed controlled-cold cohort.

The first MicroLens development ladder demonstrated a substantive tradeoff, not a promotion: M4 had the highest controlled-cold pilot NDCG@20 but weak warm performance, while M7 had the highest warm pilot NDCG@20 but weak cold performance. These single-seed validation results cannot determine the final model. Baseline qualification, equal-budget HPO, independent seeds, and comparator/margin freeze must precede locked evaluation.

The project also tests reproducibility boundaries. Publisher feature files and temporal interaction splits are hash-checked; a post-reboot audit preserved all prior receipts and checkpoints. PixelRec official publisher image features are qualified for local use, but its raw cover archive is quota-blocked. Historical content-effective timestamps are unavailable, so a strong historical-content leakage claim is not supported by the current sources.

The strongest truthful outcome may be a retained baseline or a cold/warm tradeoff. The final report must follow the evidence, including unfavorable results.
