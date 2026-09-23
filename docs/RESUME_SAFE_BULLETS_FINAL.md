# Resume-safe bullets: final variants

Use a small subset appropriate to the role. Bracketed identifiers map directly to supported final claim-ledger entries.

## Applied Scientist

- Designed and executed a protected temporal recommendation study with separate sequence and cold-capable comparators, frozen finalist selection, exact full-catalog evaluation, and paired bootstrap inference; retained the baseline when both frozen promotion gates failed. `[terminal_decision]`
- Evaluated a multimodal sequential recommender that improved the controlled-cold point estimate by `0.008940` NDCG@20, while correctly withholding a superiority claim because the paired interval crossed zero. `[locked_cold_point_improvement]`
- Quantified a `0.023619` NDCG@20 warm-item deficit against BERT4Rec and rejected promotion under a frozen noninferiority rule. `[locked_warm_noninferiority_failed]`
- Preserved a `0.007482` NDCG@20 overall secondary improvement without allowing it to override failed co-primary gates. `[locked_overall_secondary]`

## Research Scientist

- Built an auditable cold-start recommendation protocol that separated development, qualification, and protected final evidence, then closed with `RETAIN_BASELINE` when confirmatory gates failed. `[terminal_decision]`
- Used paired inference to distinguish a promising `0.008940` NDCG@20 cold point improvement from established superiority; the confidence interval crossed zero. `[locked_cold_point_improvement]`
- Identified and preserved a cold/warm tradeoff across multimodal fusion variants, including an unfavorable locked warm result rather than selecting a post hoc success metric. `[development_ladder_tradeoff]` `[locked_warm_noninferiority_failed]`
- Enforced claim discipline by treating the favorable `0.007482` overall result as secondary and keeping reliability and external replication claims unsupported. `[locked_overall_secondary]`

## ML Engineer / Applied ML

- Implemented a reproducible CUDA recommendation pipeline with temporal data contracts, exact full-catalog scoring, checkpoint and receipt hashing, canonical DuckDB/Parquet evidence, and a guarded FastAPI demonstration.
- Built a local GPU recommendation API that completed 100 sequential requests without error at `4.49 ms` median latency; reported it strictly as development demonstration evidence. `[systems_local_measurement]`
- Operationalized an auditable model-decision workflow that preserved failed runs and negative outcomes and terminated with `RETAIN_BASELINE` under frozen gates. `[terminal_decision]`
- Integrated aligned PixelRec publisher image and MiniLM text features across 82,865 items for local QA, without claiming model replication or redistribution rights. `[pixelrec_local_content_alignment]`

## Short variants

- Ran exact protected evaluation of a multimodal recommender and retained the baseline when cold uncertainty and warm noninferiority gates failed. `[terminal_decision]`
- Measured a `0.008940` cold NDCG@20 point gain without overstating superiority because the paired interval crossed zero. `[locked_cold_point_improvement]`
- Built a CUDA/FastAPI recommendation demonstration with `4.49 ms` median local sequential-request latency and explicit non-production scope. `[systems_local_measurement]`

## Prohibited rewrites

Do not convert these bullets into claims that MOSAIC:

- established controlled-cold superiority;
- retained warm quality;
- was promoted over the baseline;
- demonstrated reliable or calibrated gating;
- replicated externally;
- improved production engagement or caused user impact;
- was production ready; or
- demonstrated multi-GPU scaling.

The final evidence supports `RETAIN_BASELINE`, scoped offline observations, and a local systems demonstration only.
