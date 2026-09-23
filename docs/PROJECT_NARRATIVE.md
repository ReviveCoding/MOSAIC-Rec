# Project narrative

MOSAIC-Rec tested whether adaptive fusion of collaborative, text, image, and metadata signals could improve controlled inductive cold-item ranking without sacrificing warm-item recommendation. The study used temporal FIT, VALIDATION, QUALIFICATION, and LOCKED_FINAL roles and a fixed cohort whose collaborative FIT evidence was hidden.

The development ladder exposed a real tension: variants that favored cold content fallback tended to lose warm quality. Qualification froze BERT4Rec as the sequence comparator, a content Two-Tower as the cold comparator, and M1_T2 as the balanced MOSAIC finalist. Exact three-seed full-catalog locked scoring preserved the same pattern.

M1_T2's controlled-cold point estimate exceeded the Two-Tower by 0.008940 NDCG@20, but the paired interval crossed zero. Its warm score was 0.023619 below BERT4Rec and failed the frozen noninferiority margin. Although its secondary overall score improved, the co-primary gates required both cold superiority and warm retention. The protocol therefore closed with `RETAIN_BASELINE`.

The project also shows why preregistration details matter. Robustness and ablation family names had been frozen, but their exact interventions and implementations had not. They were marked invalid after protected access rather than repaired to fit the outcome. External inputs were prepared, but the transfer protocol was not frozen, so the conclusion remains local to MicroLens.

The engineering result is an auditable single-GPU pipeline with preserved failures, exact scoring, canonical DuckDB/Parquet evidence, paired inference, a local serving demonstration, and a claim ledger that separates supported observations from promotion, reliability, production, and causal claims.
